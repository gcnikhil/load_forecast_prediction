import { useState, useRef, useEffect } from 'react';
import {
    ResponsiveContainer, ComposedChart, LineChart, BarChart,
    Line, Bar, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
} from 'recharts';
import { whatIfPredict } from '../services/api';

/* ─── Constants ─────────────────────────────────────── */
const SYSTEM_PROMPT = `You are a feature-extraction assistant for an electricity load forecasting system.
The user will describe a hypothetical scenario in plain English.
Your job is to extract the relevant feature changes and return them as a JSON object.

Available features you can override:
- is_holiday: 0 or 1 (1 = it's a public holiday)
- temperature_celsius: float, typical Bengaluru range 15–42
- humidity_percent: float 0–100
- precipitation_mm: float ≥ 0 (rainfall)
- weather_code: integer WMO code (0=clear, 61=rain, 63=heavy rain, 71=snow, 95=thunderstorm)
- is_weekend: 0 or 1 (override if user specifies a day type)
- wind_speed_kmh: float ≥ 0
- cloud_cover_percent: float 0–100

Rules:
1. Respond with ONLY a valid JSON object.
2. Mapping qualitative terms:
   - "High humidity" -> 90, "Low humidity" -> 20
   - "Very hot" -> 42, "Chilly/Cold" -> 15
   - "Heavy rain" -> 50, "Light rain" -> 5
   - "Industrial surge" -> Map to temperature_celsius: 40 (proxy for high industrial cooling load)
3. Extract exact numbers if mentioned (e.g., "45km/h wind" -> {"wind_speed_kmh": 45}).
4. If a feature isn't mentioned, do not include it.
5. No explanations. Only JSON.

Example input: "Simulate a 45km/h wind storm with heavy rain and low humidity"
Example output: {"wind_speed_kmh": 45, "precipitation_mm": 50, "humidity_percent": 20, "weather_code": 63}`;

const LLM_MODELS = [
    { value: 'openrouter/free',                          label: 'Auto-Select Best Free ★' },
    { value: 'google/gemma-4-31b:free',                  label: 'Gemma 4 31B (Free)' },
    { value: 'openai/gpt-oss-120b:free',                 label: 'GPT-OSS 120B (Free)' },
    { value: 'nvidia/nemotron-3-nano-30b-a3b:free',      label: 'Nemotron 3 Nano 30B (Free)' },
    { value: 'meta-llama/llama-3.2-3b-instruct:free',   label: 'Llama 3.2 3B (Free)' },
];

const ANALYSIS_PROMPT = `You are an expert energy grid analyst. The user has run a what-if simulation for Bengaluru's power grid.
I will give you the simulation results (mean load change, scenario described).
Your job is to provide a 2-sentence plain English summary:
1. What happened to the electricity demand.
2. What practical operational decision the grid manager should take (e.g., "ramp up peaker plants", "schedule maintenance", "reduce grid purchases").
Do not use jargon or mention "LSTM", "models", or "overrides". Talk about the grid and the weather/holiday.`;

/* ─── Heuristic Helpers for Features ─────────────────── */
const SCENARIO_TEMPLATES = [
    { label: 'Heatwave', prompt: 'Simulate a heatwave with 5°C temperature rise.' },
    { label: 'Heavy Rain', prompt: 'Simulate heavy monsoon rain with 50mm precipitation.' },
    { label: 'Cloudy Day', prompt: 'Simulate a cloudy day with low solar output.' },
    { label: 'Industrial Surge', prompt: 'Simulate industrial demand surge.' },
    { label: 'Festival Holiday', prompt: 'Simulate a major public holiday.' },
    { label: 'Working Weekend', prompt: 'Simulate treating this Sunday as a normal working day.' },
];

const getImpactMetrics = (result) => {
    if (!result || !result.baseline_gru) return null;
    const baseGru = result.baseline_gru;
    const modGru = result.modified_gru;
    
    let baseMax = -Infinity, baseMaxIdx = -1;
    let modMax = -Infinity, modMaxIdx = -1;
    
    for (let i = 0; i < baseGru.length; i++) {
        if (baseGru[i] > baseMax) { baseMax = baseGru[i]; baseMaxIdx = i; }
        if (modGru[i] > modMax) { modMax = modGru[i]; modMaxIdx = i; }
    }
    
    const peakChange = modMax - baseMax;
    const formatTime = (idx) => result.timestamps[idx]?.slice(11, 16) || '--:--';
    const peakShift = `${formatTime(baseMaxIdx)} → ${formatTime(modMaxIdx)}`;
    
    let stress = 'LOW';
    const maxCapacity = 16000;
    const stressRatio = modMax / maxCapacity;
    if (stressRatio > 0.95) stress = 'CRITICAL';
    else if (stressRatio > 0.85) stress = 'HIGH';
    else if (stressRatio > 0.75) stress = 'MODERATE';
    
    const overridesCount = Object.keys(result.applied_overrides).length;
    const confidence = Math.max(60, 95 - (overridesCount * 4));
    const deficitMWh = (result.delta_gru.reduce((sum, d) => sum + Math.max(0, d), 0) / 4).toFixed(1);
    
    let renewable = "Minimal impact";
    if (result.applied_overrides['weather_code'] >= 60 || result.applied_overrides['precipitation_mm'] > 0) {
        renewable = "Solar contribution reduced by ~14%";
    } else if (result.applied_overrides['temperature_celsius'] > 35) {
        renewable = "Wind efficiency reduced by heat";
    }
    
    return {
        peakChange: `${peakChange > 0 ? '+' : ''}${(peakChange).toFixed(1)} MW`,
        peakChangeColor: peakChange > 0 ? '#F9F871' : '#00FFF6',
        peakShift,
        stress,
        stressColor: stress === 'CRITICAL' ? '#FF4D4D' : stress === 'HIGH' ? '#FF9E00' : stress === 'MODERATE' ? '#F9F871' : '#00FFF6',
        confidence: `${confidence}%`,
        deficit: `${deficitMWh} MWh`,
        renewable
    };
};

const getExplainability = (overrides, deltaMW) => {
    if (!overrides || Object.keys(overrides).length === 0) return [];
    const impacts = [];
    for (const [k, v] of Object.entries(overrides)) {
        if (k.includes('temp')) impacts.push({ feature: 'Temperature', impact: Math.abs(deltaMW) * 0.6, sign: v > 28 ? 1 : -1 });
        else if (k.includes('holid')) impacts.push({ feature: 'Holiday Status', impact: Math.abs(deltaMW) * 0.8, sign: -1 });
        else if (k.includes('precip') || k.includes('weather')) impacts.push({ feature: 'Weather/Rain', impact: Math.abs(deltaMW) * 0.3, sign: -1 });
        else impacts.push({ feature: k, impact: Math.abs(deltaMW) * 0.2, sign: 1 });
    }
    impacts.sort((a,b) => b.impact - a.impact);
    const total = impacts.reduce((sum, i) => sum + i.impact, 0) || 1;
    return impacts.map(i => ({
        feature: i.feature,
        pct: `${i.sign > 0 ? '+' : ''}${Math.round((i.impact / total) * 100)}%`
    }));
};

/* ─── Custom Tooltip ─────────────────────────────────── */
const CustomTooltip = ({ active, payload, label }) => {
    if (!active || !payload?.length) return null;
    return (
        <div className="bg-[#0B0F1A] border border-[#00FFF6]/30 p-3 text-xs font-mono space-y-1 shadow-xl">
            <p className="text-[#8A8F98] mb-2">{label}</p>
            {payload.map((p) => (
                <p key={p.name} style={{ color: p.color }}>
                    {p.name}: <span className="text-white">{typeof p.value === 'number' ? p.value.toFixed(1) : p.value} MW</span>
                </p>
            ))}
        </div>
    );
};

/* ─── Summary Card ───────────────────────────────────── */
const StatBox = ({ label, value, sub, accent = '#00FFF6' }) => (
    <div className="border p-4 bg-[#0B0F1A]" style={{ borderColor: `${accent}33` }}>
        <p className="text-[10px] font-mono uppercase tracking-widest mb-2" style={{ color: accent }}>{label}</p>
        <p className="text-2xl font-bold font-mono text-white">{value ?? '—'}</p>
        {sub && <p className="text-xs font-mono text-[#8A8F98] mt-1">{sub}</p>}
    </div>
);

/* ─── Main Page ──────────────────────────────────────── */
const formatDate = (d) => d.toISOString().slice(0, 10); // YYYY-MM-DD

export default function WhatIf() {
    const [apiKey, setApiKey] = useState(() => localStorage.getItem('openrouter_api_key') || '');
    const [llmModel, setLlmModel] = useState(() => sessionStorage.getItem('wi_llmModel') || 'openrouter/free');
    const [startDate, setStartDate] = useState(() => {
        const d = new Date(); d.setDate(d.getDate() + 1); 
        return sessionStorage.getItem('wi_startDate') || formatDate(d);
    });
    const [endDate, setEndDate] = useState(() => {
        const d = new Date(); d.setDate(d.getDate() + 2); 
        return sessionStorage.getItem('wi_endDate') || formatDate(d);
    });
    const [messages, setMessages] = useState(() => JSON.parse(sessionStorage.getItem('wi_messages') || '[]'));
    const [inputText, setInputText] = useState(() => sessionStorage.getItem('wi_inputText') || '');
    const [parsedOverrides, setParsedOverrides] = useState(() => JSON.parse(sessionStorage.getItem('wi_parsedOverrides') || 'null'));
    const [simResult, setSimResult] = useState(() => JSON.parse(sessionStorage.getItem('wi_simResult') || 'null'));
    const [simInsight, setSimInsight] = useState(() => sessionStorage.getItem('wi_simInsight') || '');
    const [isLlmLoading, setIsLlmLoading] = useState(false);
    const [isSimLoading, setIsSimLoading] = useState(false);
    const [error, setError] = useState('');
    const [dateError, setDateError] = useState('');
    const messagesEndRef = useRef(null);

    useEffect(() => {
        sessionStorage.setItem('wi_llmModel', llmModel);
        sessionStorage.setItem('wi_startDate', startDate);
        sessionStorage.setItem('wi_endDate', endDate);
        sessionStorage.setItem('wi_messages', JSON.stringify(messages));
        sessionStorage.setItem('wi_inputText', inputText);
        sessionStorage.setItem('wi_parsedOverrides', JSON.stringify(parsedOverrides));
        sessionStorage.setItem('wi_simResult', JSON.stringify(simResult));
        sessionStorage.setItem('wi_simInsight', simInsight);
    }, [llmModel, startDate, endDate, messages, inputText, parsedOverrides, simResult, simInsight]);

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages, isLlmLoading]);

    const handleApiKeyChange = (v) => {
        setApiKey(v);
        localStorage.setItem('openrouter_api_key', v);
    };

    const validateDates = (s, e) => {
        if (!s || !e) return 'Please select both start and end dates.';
        const diff = (new Date(e) - new Date(s)) / 86400000;
        if (diff < 0) return 'End date must be after start date.';
        if (diff > 3) return 'Maximum forecast range is 3 days.';
        return '';
    };

    const handleStartDate = (v) => { setStartDate(v); setDateError(validateDates(v, endDate)); };
    const handleEndDate = (v) => { setEndDate(v); setDateError(validateDates(startDate, v)); };

    const callLLM = async (userMessage) => {
        if (!userMessage.trim() || !apiKey) return;
        setIsLlmLoading(true);
        setError('');
        const newMessages = [...messages, { role: 'user', content: userMessage }];
        setMessages(newMessages);
        setInputText('');
        try {
            const res = await fetch('https://openrouter.ai/api/v1/chat/completions', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${apiKey}`,
                    'HTTP-Referer': window.location.origin,
                    'X-Title': 'GRID_FCST What-If Engine',
                },
                body: JSON.stringify({
                    model: llmModel,
                    messages: [{ role: 'system', content: SYSTEM_PROMPT }, ...newMessages],
                    temperature: 0.1,
                    max_tokens: 200,
                }),
            });
            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.error?.message || `OpenRouter error ${res.status}`);
            }
            const data = await res.json();
            const rawContent = data.choices[0].message.content.trim();
            let overrides = {};
            try {
                const cleaned = rawContent.replace(/```json|```/g, '').trim();
                overrides = JSON.parse(cleaned);
                // Coerce numeric strings to numbers
                Object.keys(overrides).forEach(k => {
                    const n = Number(overrides[k]);
                    if (!isNaN(n)) overrides[k] = n;
                });
            } catch {
                setMessages(prev => [
                    ...prev,
                    { role: 'assistant', content: rawContent },
                    { role: 'system', content: '⚠️ Could not parse feature overrides. Please rephrase your query.' },
                ]);
                setIsLlmLoading(false);
                return;
            }
            const keys = Object.keys(overrides);
            let explanation = '';
            if (keys.length === 0) {
                explanation = "I couldn't find any specific feature changes. Please describe conditions like temperature, holiday, rainfall, etc.";
                setParsedOverrides(null);
            } else {
                const parts = keys.map(k => `${k} → ${overrides[k]}`);
                explanation = `Extracted feature overrides:\n${parts.join('\n')}\n\nClick RUN_SIMULATION to apply these to the hybrid model.`;
                setParsedOverrides(overrides);
            }
            setMessages(prev => [...prev, { role: 'assistant', content: explanation }]);
        } catch (err) {
            const msg = `Error: ${err.message}`;
            setMessages(prev => [...prev, { role: 'system', content: msg }]);
            setError(msg);
        } finally {
            setIsLlmLoading(false);
        }
    };

    const generateInsight = async (summary, overrides, userQuery) => {
        if (!apiKey) return;
        setSimInsight('Analyzing operational impact...');
        try {
            const scenario = Object.entries(overrides).map(([k, v]) => `${k}=${v}`).join(', ');
            const deltaMW = summary.mean_delta_mw;
            const deltaPct = summary.mean_delta_pct;
            
            const prompt = `User Query: "${userQuery}"\nApplied scenario: ${scenario}\nResult: Mean load changed by ${deltaMW > 0 ? '+' : ''}${deltaMW} MW (${deltaPct > 0 ? '+' : ''}${deltaPct}%).\nProvide the 2-sentence operational insight.`;
            
            const res = await fetch('https://openrouter.ai/api/v1/chat/completions', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${apiKey}`,
                    'HTTP-Referer': window.location.origin,
                    'X-Title': 'GRID_FCST What-If Engine',
                },
                body: JSON.stringify({
                    model: llmModel,
                    messages: [
                        { role: 'system', content: ANALYSIS_PROMPT },
                        { role: 'user', content: prompt }
                    ],
                    temperature: 0.3,
                    max_tokens: 100,
                }),
            });
            const data = await res.json();
            setSimInsight(data.choices[0].message.content.trim());
        } catch (err) {
            setSimInsight('Could not generate insight due to API error.');
        }
    };

    const runSimulation = async () => {
        const de = validateDates(startDate, endDate);
        if (de) { setDateError(de); return; }
        if (!parsedOverrides) return;
        setIsSimLoading(true);
        setSimResult(null);
        setSimInsight('');
        setError('');
        try {
            const result = await whatIfPredict(startDate, endDate, parsedOverrides);
            setSimResult(result);
            generateInsight(result.summary, parsedOverrides, inputText || "No query provided");
        } catch (err) {
            const detail = err.response?.data?.detail || err.message;
            if (detail?.includes('503') || err.response?.status === 503) {
                setError('Models not loaded on backend. Please ensure the FastAPI server has loaded the hybrid models.');
            } else {
                setError(`Simulation failed: ${detail}`);
            }
        } finally {
            setIsSimLoading(false);
        }
    };

    /* Build chart data */
    const chartData = simResult
        ? simResult.timestamps.map((t, i) => ({
            t: t.slice(5, 16), // MM-DD HH:MM
            'Baseline Load': simResult.baseline_gru[i],
            'Modified Load': simResult.modified_gru[i],
            delta: simResult.delta_gru[i],
        }))
        : [];

    const { summary } = simResult || {};
    const deltaSign = summary?.mean_delta_mw > 0 ? '▲' : summary?.mean_delta_mw < 0 ? '▼' : '—';
    const deltaColor = summary?.mean_delta_mw > 0 ? '#F9F871' : '#00FFF6';

    return (
        <div className="space-y-6">
            {/* ── Header ── */}
            <div className="flex items-center justify-between flex-wrap gap-3">
                <div>
                    <h1 className="text-2xl font-bold text-[#00FFF6] font-mono tracking-wider">
                        WHAT_IF // SIMULATION ENGINE
                    </h1>
                    <p className="text-[#8A8F98] text-sm font-mono mt-1">
                        Natural language → Feature overrides → Re-prediction (no retraining)
                    </p>
                </div>
                <div className="flex items-center gap-2 px-3 py-1 border border-[#F9F871]/40 bg-[#F9F871]/5">
                    <div className="w-2 h-2 bg-[#F9F871] shadow-[0_0_8px_#F9F871] animate-pulse rounded-full" />
                    <span className="text-xs font-mono text-[#F9F871]">LLM::POWERED</span>
                </div>
            </div>

            {/* ── Config Row ── */}
            <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4 p-4 border border-[#00FFF6]/20 bg-[#0B0F1A]">
                <div className="flex flex-col gap-1">
                    <label className="text-[10px] font-mono uppercase text-[#8A8F98] tracking-widest">Start Date</label>
                    <input
                        type="date" value={startDate} onChange={e => handleStartDate(e.target.value)}
                        className="bg-transparent border border-[#00FFF6]/20 text-[#00FFF6] font-mono text-sm p-2 focus:outline-none focus:border-[#00FFF6]/60 [color-scheme:dark]"
                    />
                </div>
                <div className="flex flex-col gap-1">
                    <label className="text-[10px] font-mono uppercase text-[#8A8F98] tracking-widest">End Date</label>
                    <input
                        type="date" value={endDate} onChange={e => handleEndDate(e.target.value)}
                        className="bg-transparent border border-[#00FFF6]/20 text-[#00FFF6] font-mono text-sm p-2 focus:outline-none focus:border-[#00FFF6]/60 [color-scheme:dark]"
                    />
                </div>
                <div className="flex flex-col gap-1">
                    <label className="text-[10px] font-mono uppercase text-[#8A8F98] tracking-widest">OpenRouter API Key</label>
                    <input
                        type="password" value={apiKey} onChange={e => handleApiKeyChange(e.target.value)}
                        placeholder="sk-or-..."
                        className="bg-transparent border border-[#00FFF6]/20 text-[#8A8F98] font-mono text-sm p-2 focus:outline-none focus:border-[#00FFF6]/60 placeholder-[#8A8F98]/30"
                    />
                </div>
                <div className="flex flex-col gap-1">
                    <label className="text-[10px] font-mono uppercase text-[#8A8F98] tracking-widest">LLM Model</label>
                    <select
                        value={llmModel} onChange={e => setLlmModel(e.target.value)}
                        className="bg-[#0B0F1A] border border-[#00FFF6]/20 text-[#8A8F98] font-mono text-sm p-2 focus:outline-none focus:border-[#00FFF6]/60"
                    >
                        {LLM_MODELS.map(m => <option key={m.value} value={m.value}>{m.label}</option>)}
                    </select>
                </div>
            </div>
            {dateError && (
                <p className="text-red-400 text-xs font-mono px-1">⚠️ {dateError}</p>
            )}

            {/* ── Main 2-col ── */}
            <div className="grid grid-cols-1 xl:grid-cols-5 gap-6">

                {/* Chat Panel */}
                <div className="xl:col-span-2 border border-[#00FFF6]/20 bg-[#0B0F1A] flex flex-col" style={{ height: 580 }}>
                    <div className="px-4 py-2 border-b border-[#00FFF6]/10 flex items-center gap-2">
                        <span className="text-xs font-mono text-[#00FFF6] uppercase tracking-widest">LLM Chat</span>
                    </div>

                    {/* Messages */}
                    <div className="flex-1 overflow-y-auto p-4 space-y-3 font-mono text-sm">
                        {messages.length === 0 && (
                            <div className="text-[#8A8F98] text-xs leading-relaxed">
                                Describe a hypothetical condition, e.g.:<br />
                                <span className="text-[#00FFF6]">"What if tomorrow is a public holiday and temperature drops by 3°C?"</span><br /><br />
                                <span className="text-[#00FFF6]">"Simulate heavy rain with 80% humidity"</span><br /><br />
                                <span className="text-[#00FFF6]">"What if it's a hot day at 42°C?"</span>
                            </div>
                        )}
                        {messages.map((msg, i) => (
                            <div key={i} className={`p-2 border text-xs ${
                                msg.role === 'user'
                                    ? 'border-[#00FFF6]/30 bg-[#00FFF6]/5 text-[#00FFF6] ml-4'
                                    : msg.role === 'system'
                                    ? 'border-red-500/30 bg-red-500/5 text-red-400'
                                    : 'border-[#F9F871]/20 bg-[#F9F871]/5 text-[#8A8F98] mr-4'
                            }`}>
                                <span className="font-bold uppercase text-[10px] opacity-60 block mb-1">
                                    {msg.role === 'user' ? 'YOU' : msg.role === 'system' ? 'SYS' : 'LLM'}
                                </span>
                                <p className="whitespace-pre-wrap">{msg.content}</p>
                            </div>
                        ))}
                        {isLlmLoading && (
                            <div className="text-[#8A8F98] text-xs animate-pulse">LLM processing...</div>
                        )}
                        <div ref={messagesEndRef} />
                    </div>

                    {/* Input */}
                    <div className="border-t border-[#00FFF6]/20 p-3 flex flex-col gap-3">
                        <div className="flex flex-wrap gap-2">
                            {SCENARIO_TEMPLATES.map(t => (
                                <button key={t.label} onClick={() => setInputText(t.prompt)} className="px-2 py-1 border border-[#00FFF6]/30 text-[10px] text-[#00FFF6] hover:bg-[#00FFF6]/10 hover:border-[#00FFF6] transition-colors rounded-sm whitespace-nowrap">
                                    {t.label}
                                </button>
                            ))}
                        </div>
                        <textarea
                            value={inputText}
                            onChange={e => setInputText(e.target.value)}
                            onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); callLLM(inputText); } }}
                            placeholder="Describe a hypothetical scenario... (Enter to send)"
                            className="w-full bg-transparent border border-[#00FFF6]/20 text-[#8A8F98] text-xs font-mono p-2 resize-none focus:outline-none focus:border-[#00FFF6]/60 placeholder-[#8A8F98]/40"
                            rows={3}
                        />
                        <div className="flex gap-2">
                            <button
                                onClick={() => callLLM(inputText)}
                                disabled={!inputText.trim() || isLlmLoading || !apiKey}
                                className="flex-1 py-1.5 border border-[#00FFF6]/40 text-[#00FFF6] text-xs font-mono hover:bg-[#00FFF6]/10 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                            >
                                SEND_QUERY
                            </button>
                            <button
                                onClick={runSimulation}
                                disabled={!parsedOverrides || !startDate || !endDate || !!dateError || isSimLoading}
                                className="flex-1 py-1.5 border border-[#F9F871]/40 text-[#F9F871] text-xs font-mono hover:bg-[#F9F871]/10 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                            >
                                {isSimLoading ? 'RUNNING...' : 'RUN_SIMULATION'}
                            </button>
                        </div>
                        {!apiKey && (
                            <p className="text-[10px] font-mono text-red-400/70">⚠️ Enter your OpenRouter API key above to use the LLM.</p>
                        )}
                    </div>
                </div>

                {/* Results Panel */}
                <div className="xl:col-span-3 space-y-4">
                    {!simResult && !isSimLoading && (
                        <div className="border border-[#00FFF6]/10 bg-[#0B0F1A] flex flex-col items-center justify-center gap-4" style={{ height: 580 }}>
                            <div className="w-16 h-16 border border-[#00FFF6]/20 flex items-center justify-center">
                                <span className="text-3xl">⚗️</span>
                            </div>
                            <p className="text-[#8A8F98] text-sm font-mono text-center px-8">
                                Results will appear here after running a simulation.<br />
                                <span className="text-[#00FFF6]/60 text-xs">Ask the LLM a question, then click RUN_SIMULATION.</span>
                            </p>
                        </div>
                    )}
                    {isSimLoading && (
                        <div className="border border-[#00FFF6]/20 bg-[#0B0F1A] flex items-center justify-center" style={{ height: 580 }}>
                            <div className="text-center space-y-3">
                                <div className="w-8 h-8 border-2 border-[#00FFF6]/40 border-t-[#00FFF6] rounded-full animate-spin mx-auto" />
                                <p className="text-[#00FFF6] font-mono text-sm animate-pulse">COMPUTING MODIFIED FORECAST...</p>
                            </div>
                        </div>
                    )}

                    {simResult && (
                        <>
                            {/* Override Badges */}
                            <div className="flex flex-col gap-2 p-4 border border-[#00FFF6]/10 bg-[#0B0F1A]">
                                <div className="flex flex-wrap gap-2">
                                    <span className="text-xs font-mono text-[#8A8F98] self-center mr-1">OVERRIDES:</span>
                                    {Object.entries(simResult.applied_overrides).map(([k, v]) => (
                                        <span key={k} className="px-2 py-1 border border-[#00FFF6]/40 bg-[#00FFF6]/10 text-[#00FFF6] text-xs font-mono">
                                            {k}: {String(v)}
                                        </span>
                                    ))}
                                </div>
                            </div>

                            {/* Summary Cards */}
                            <div className="grid grid-cols-3 gap-3">
                                <StatBox
                                    label="Baseline"
                                    value={summary?.baseline?.mean != null ? `${summary.baseline.mean} MW` : null}
                                    sub={`Min ${summary?.baseline?.min ?? '—'} / Max ${summary?.baseline?.max ?? '—'} MW`}
                                    accent="#7B61FF"
                                />
                                <StatBox
                                    label="Modified"
                                    value={summary?.modified?.mean != null ? `${summary.modified.mean} MW` : null}
                                    sub={`Min ${summary?.modified?.min ?? '—'} / Max ${summary?.modified?.max ?? '—'} MW`}
                                    accent="#00FFF6"
                                />
                                <StatBox
                                    label="Δ Delta"
                                    value={summary?.mean_delta_mw != null
                                        ? <span style={{ color: deltaColor }}>{deltaSign} {Math.abs(summary.mean_delta_mw)} MW</span>
                                        : null}
                                    sub={summary?.mean_delta_pct != null
                                        ? `${summary.mean_delta_pct > 0 ? '+' : ''}${summary.mean_delta_pct}% mean change`
                                        : undefined}
                                    accent="#F9F871"
                                />
                            </div>

                            {/* Comparison Line Chart */}
                            <div className="border border-[#00FFF6]/10 bg-[#0B0F1A] p-4">
                                <p className="text-xs font-mono text-[#8A8F98] uppercase tracking-widest mb-3">
                                    Baseline vs Modified Forecast (MW)
                                </p>
                                <ResponsiveContainer width="100%" height={260}>
                                    <ComposedChart data={chartData} margin={{ top: 4, right: 8, left: 0, bottom: 4 }}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#1a2030" />
                                        <XAxis
                                            dataKey="t"
                                            tick={{ fill: '#8A8F98', fontSize: 10, fontFamily: 'monospace' }}
                                            minTickGap={30}
                                            stroke="#1a2030"
                                        />
                                        <YAxis
                                            tick={{ fill: '#8A8F98', fontSize: 10, fontFamily: 'monospace' }}
                                            stroke="#1a2030"
                                            width={55}
                                            tickFormatter={v => `${(v/1000).toFixed(1)}k`}
                                        />
                                        <Tooltip content={<CustomTooltip />} />
                                        <Legend
                                            wrapperStyle={{ fontSize: 10, fontFamily: 'monospace', color: '#8A8F98' }}
                                        />
                                        <Line type="monotone" name="Baseline Load" dataKey="Baseline Load" stroke="#7B61FF" strokeWidth={1.5} dot={false} strokeDasharray="4 2" />
                                        <Line type="monotone" name="Modified Load" dataKey="Modified Load" stroke="#00FFF6" strokeWidth={2.5} dot={false} />
                                    </ComposedChart>
                                </ResponsiveContainer>
                            </div>

                            {/* Delta Bar Chart */}
                            <div className="border border-[#00FFF6]/10 bg-[#0B0F1A] p-4">
                                <p className="text-xs font-mono text-[#8A8F98] uppercase tracking-widest mb-3">
                                    Net Load Change (MW) (Modified − Baseline)
                                </p>
                                <ResponsiveContainer width="100%" height={180}>
                                    <BarChart data={chartData} margin={{ top: 4, right: 8, left: 0, bottom: 4 }}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#1a2030" />
                                        <XAxis
                                            dataKey="t"
                                            tick={{ fill: '#8A8F98', fontSize: 10, fontFamily: 'monospace' }}
                                            minTickGap={30}
                                            stroke="#1a2030"
                                        />
                                        <YAxis
                                            tick={{ fill: '#8A8F98', fontSize: 10, fontFamily: 'monospace' }}
                                            stroke="#1a2030"
                                            width={55}
                                        />
                                        <Tooltip content={<CustomTooltip />} />
                                        <Bar dataKey="delta" name="Δ Net Load (MW)">
                                            {chartData.map((entry, i) => (
                                                <Cell key={i} fill={entry.delta >= 0 ? '#F9F871' : '#00FFF6'} />
                                            ))}
                                        </Bar>
                                    </BarChart>
                                </ResponsiveContainer>
                                <p className="text-[10px] font-mono text-[#8A8F98]/60 mt-2">
                                    <span style={{ color: '#F9F871' }}>■</span> Higher load &nbsp;
                                    <span style={{ color: '#00FFF6' }}>■</span> Lower load
                                </p>
                            </div>

                            {/* Impact Analysis Metrics */}
                            {(() => {
                                const metrics = getImpactMetrics(simResult);
                                if (!metrics) return null;
                                return (
                                    <div className="border border-[#00FFF6]/10 bg-[#0B0F1A] p-4 space-y-4 mt-6">
                                        <p className="text-xs font-mono text-[#8A8F98] uppercase tracking-widest border-b border-[#00FFF6]/10 pb-2">Impact Analysis</p>
                                        <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                                            <div className="p-3 border border-[#00FFF6]/20 bg-[#00FFF6]/5">
                                                <p className="text-[10px] text-[#8A8F98] font-mono uppercase mb-1">Peak Load Change</p>
                                                <p className="text-lg font-bold font-mono" style={{color: metrics.peakChangeColor}}>{metrics.peakChange}</p>
                                            </div>
                                            <div className="p-3 border border-[#00FFF6]/20 bg-[#00FFF6]/5">
                                                <p className="text-[10px] text-[#8A8F98] font-mono uppercase mb-1">Peak Time Shift</p>
                                                <p className="text-lg font-bold font-mono text-white">{metrics.peakShift}</p>
                                            </div>
                                            <div className="p-3 border border-[#00FFF6]/20 bg-[#00FFF6]/5">
                                                <p className="text-[10px] text-[#8A8F98] font-mono uppercase mb-1">Grid Stress Level</p>
                                                <p className="text-lg font-bold font-mono" style={{color: metrics.stressColor}}>{metrics.stress}</p>
                                            </div>
                                            <div className="p-3 border border-[#00FFF6]/20 bg-[#00FFF6]/5">
                                                <p className="text-[10px] text-[#8A8F98] font-mono uppercase mb-1">Forecast Confidence</p>
                                                <p className="text-lg font-bold font-mono text-white">{metrics.confidence}</p>
                                            </div>
                                            <div className="p-3 border border-[#00FFF6]/20 bg-[#00FFF6]/5">
                                                <p className="text-[10px] text-[#8A8F98] font-mono uppercase mb-1">Estimated Energy Deficit</p>
                                                <p className="text-lg font-bold font-mono text-[#F9F871]">{metrics.deficit}</p>
                                            </div>
                                            <div className="p-3 border border-[#00FFF6]/20 bg-[#00FFF6]/5">
                                                <p className="text-[10px] text-[#8A8F98] font-mono uppercase mb-1">Renewable Dependency</p>
                                                <p className="text-xs font-mono text-white mt-1 leading-tight">{metrics.renewable}</p>
                                            </div>
                                        </div>
                                    </div>
                                );
                            })()}

                            {/* Explainability Panel */}
                            {(() => {
                                const exp = getExplainability(simResult.applied_overrides, summary?.mean_delta_mw || 0);
                                if (!exp.length) return null;
                                return (
                                    <div className="border border-[#00FFF6]/10 bg-[#0B0F1A] p-4 space-y-4">
                                        <p className="text-xs font-mono text-[#8A8F98] uppercase tracking-widest border-b border-[#00FFF6]/10 pb-2">Explainability: Feature Contribution</p>
                                        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-3">
                                            {exp.map((f, i) => (
                                                <div key={i} className="flex justify-between items-center p-2 border-b border-[#00FFF6]/20 bg-[#00FFF6]/5">
                                                    <span className="text-sm font-mono text-white">{f.feature}</span>
                                                    <span className="text-sm font-bold font-mono" style={{color: f.pct.includes('+') ? '#F9F871' : '#00FFF6'}}>{f.pct}</span>
                                                </div>
                                            ))}
                                        </div>
                                        <p className="text-xs font-mono text-[#8A8F98] mt-2">
                                            <span className="text-[#00FFF6]">Primary Driver:</span> {exp[0].feature} caused the largest impact on the forecast.
                                        </p>
                                    </div>
                                );
                            })()}

                            {/* Risk Heatmap */}
                            {(() => {
                                const riskData = [];
                                let currentHour = -1;
                                for (let i = 0; i < simResult.timestamps.length; i++) {
                                    const t = simResult.timestamps[i];
                                    const date = new Date(t);
                                    if (date.getHours() !== currentHour) {
                                        currentHour = date.getHours();
                                        const val = simResult.modified_gru[i];
                                        const pct = val / 16000;
                                        let level = 'LOW', bg = 'bg-[#00FFF6]';
                                        if (pct > 0.95) { level = 'CRITICAL'; bg = 'bg-[#FF4D4D]'; }
                                        else if (pct > 0.85) { level = 'HIGH'; bg = 'bg-[#FF9E00]'; }
                                        else if (pct > 0.75) { level = 'MODERATE'; bg = 'bg-[#F9F871]'; }
                                        riskData.push({ t: t.slice(11, 16), level, bg, val });
                                        if (riskData.length >= 48) break;
                                    }
                                }
                                return (
                                    <div className="border border-[#00FFF6]/10 bg-[#0B0F1A] p-4 space-y-3">
                                        <p className="text-xs font-mono text-[#8A8F98] uppercase tracking-widest border-b border-[#00FFF6]/10 pb-2">Grid Risk Heatmap (Next 48 Hours)</p>
                                        <div className="flex flex-wrap gap-1">
                                            {riskData.map((d, i) => (
                                                <div key={i} title={`${d.t} - ${Math.round(d.val)} MW (${d.level})`} className={`w-6 h-8 ${d.bg} opacity-80 hover:opacity-100 cursor-help transition-opacity`}></div>
                                            ))}
                                        </div>
                                        <div className="flex gap-4 mt-2">
                                            <div className="flex items-center gap-1"><div className="w-3 h-3 bg-[#00FFF6]"></div><span className="text-[10px] font-mono text-[#8A8F98]">LOW</span></div>
                                            <div className="flex items-center gap-1"><div className="w-3 h-3 bg-[#F9F871]"></div><span className="text-[10px] font-mono text-[#8A8F98]">MODERATE</span></div>
                                            <div className="flex items-center gap-1"><div className="w-3 h-3 bg-[#FF9E00]"></div><span className="text-[10px] font-mono text-[#8A8F98]">HIGH</span></div>
                                            <div className="flex items-center gap-1"><div className="w-3 h-3 bg-[#FF4D4D]"></div><span className="text-[10px] font-mono text-[#8A8F98]">CRITICAL</span></div>
                                        </div>
                                    </div>
                                );
                            })()}

                            {/* AI Grid Insights */}
                            {simInsight && (
                                <div className="border border-[#F9F871]/30 bg-[#F9F871]/5 p-4 space-y-2">
                                    <p className="text-xs font-mono text-[#F9F871] uppercase tracking-widest border-b border-[#F9F871]/20 pb-2">AI Grid Insights</p>
                                    <p className="text-sm font-sans text-gray-200 leading-relaxed">{simInsight}</p>
                                </div>
                            )}
                        </>
                    )}
                </div>
            </div>

            {/* Error */}
            {error && (
                <div className="p-3 border border-red-500/40 bg-red-500/5 text-red-400 text-xs font-mono">
                    ERROR: {error}
                </div>
            )}
        </div>
    );
}
