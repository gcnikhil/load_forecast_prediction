import { useState, useEffect } from 'react';
import { useData } from '../contexts/DataContext';
import {
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
    LineChart, Line, AreaChart, Area, PieChart, Pie, Cell, ReferenceDot, LabelList
} from 'recharts';
import { predictLoad, getModelMetrics } from '../services/api';
import { 
    TrendingUp, Activity, AlertTriangle, Target, Award, Clock, 
    BarChart3, Cpu, Database, ArrowUp, ArrowDown
} from 'lucide-react';

const Analytics = () => {
    const { data: sharedData, setData: setSharedData, metrics: sharedMetrics, setMetrics: setSharedMetrics } = useData();
    const [data, setData] = useState(sharedData);
    const [loading, setLoading] = useState(!sharedData);
    const [error, setError] = useState(null);
    const [metrics, setMetrics] = useState(sharedMetrics);

    // Fix 20: Don't block on missing metrics — render page if sharedData exists
    useEffect(() => {
        if (sharedData) {
            setData(sharedData);
            setLoading(false);
        }
        if (sharedMetrics) {
            setMetrics(sharedMetrics);
            if (sharedData) return; // both available — skip fetching
        }

        const fetchMetrics = async () => {
            try {
                const metricsData = await getModelMetrics();
                setMetrics(metricsData);
                setSharedMetrics(metricsData);
            } catch (err) {
                console.error('Metrics fetch failed:', err);
                // non-fatal — page renders without metrics
            }
        };

        if (!sharedData) {
            const fetchAll = async () => {
                try {
                    const today = new Date();
                    const yesterday = new Date(today);
                    yesterday.setDate(today.getDate() - 1);
                    const formatDate = (d) => d.toISOString().split('T')[0];
                    const [result, metricsData] = await Promise.all([
                        predictLoad(formatDate(yesterday), formatDate(today)),
                        getModelMetrics()
                    ]);
                    if (!result || !result.loads_lightgbm_lstm || !result.loads_lightgbm_gru) {
                        throw new Error('Invalid data format');
                    }
                    setData(result);
                    setMetrics(metricsData);
                    setSharedData(result);
                    setSharedMetrics(metricsData);
                } catch (err) {
                    console.error('Analytics error:', err);
                    setError(err.message);
                } finally {
                    setLoading(false);
                }
            };
            fetchAll();
        } else {
            fetchMetrics();
        }
    }, [sharedData, sharedMetrics, setSharedData, setSharedMetrics]);

    if (loading) return (
        <div className="flex items-center justify-center min-h-[60vh] text-[#00FFF6]">
            <Cpu className="w-8 h-8 mr-3 animate-spin" /> Loading Analytics...
        </div>
    );

    if (error || !data) return (
        <div className="flex flex-col items-center justify-center min-h-[60vh] text-amber-500">
            <AlertTriangle className="w-12 h-12 mb-4" />
            <p className="text-slate-400">{error || "No data available"}</p>
        </div>
    );

    const lstmLoads = data.loads_lightgbm_lstm;
    const gruLoads = data.loads_lightgbm_gru;
    const timestamps = data.timestamps;

    // Fix 16: Filter null/undefined/NaN before Math.max/min to avoid -Infinity/NaN on stats cards
    const cleanLstm = lstmLoads.filter(v => v !== null && v !== undefined && !isNaN(v));
    const cleanGru  = gruLoads.filter(v => v !== null && v !== undefined && !isNaN(v));

    const lstmMax = cleanLstm.length ? Math.max(...cleanLstm) : 0;
    const lstmMin = cleanLstm.length ? Math.min(...cleanLstm) : 0;
    const lstmAvg = cleanLstm.length ? cleanLstm.reduce((a, b) => a + b, 0) / cleanLstm.length : 0;
    const gruMax  = cleanGru.length  ? Math.max(...cleanGru)  : 0;
    const gruMin  = cleanGru.length  ? Math.min(...cleanGru)  : 0;
    const gruAvg  = cleanGru.length  ? cleanGru.reduce((a, b) => a + b, 0) / cleanGru.length : 0;

    const hourlyMap = {};
    timestamps.forEach((ts, i) => {
        const hour = ts.split(' ')[1]?.split(':')[0] || '00';
        if (!hourlyMap[hour]) hourlyMap[hour] = { lstm: [], gru: [] };
        hourlyMap[hour].lstm.push(lstmLoads[i]);
        hourlyMap[hour].gru.push(gruLoads[i]);
    });

    const hourlyChartData = Object.keys(hourlyMap)
        .sort((a, b) => parseInt(a) - parseInt(b))
        .map(hour => ({
            hour: `${hour}:00`,
            LSTM: Math.round(hourlyMap[hour].lstm.reduce((a, b) => a + b, 0) / hourlyMap[hour].lstm.length),
            GRU: Math.round(hourlyMap[hour].gru.reduce((a, b) => a + b, 0) / hourlyMap[hour].gru.length)
        }));

    const comparisonData = [
        { metric: 'Peak', LSTM: Math.round(lstmMax), GRU: Math.round(gruMax) },
        { metric: 'Min', LSTM: Math.round(lstmMin), GRU: Math.round(gruMin) },
        { metric: 'Average', LSTM: Math.round(lstmAvg), GRU: Math.round(gruAvg) }
    ];

    const distributionMap = {};
    [...lstmLoads, ...gruLoads].forEach(load => {
        const bucket = Math.floor(load / 500) * 500;
        const key = `${(bucket/1000).toFixed(1)}-${((bucket+500)/1000).toFixed(1)}k`;
        distributionMap[key] = (distributionMap[key] || 0) + 1;
    });
    const distributionData = Object.entries(distributionMap)
        .map(([range, count]) => ({ range, count }))
        .sort((a, b) => parseFloat(a.range) - parseFloat(b.range));

    const varianceData = timestamps
        .filter((_, i) => i % 30 === 0)
        .map((ts, idx) => {
            const i = idx * 30;
            const lv = lstmLoads[i]; const gv = gruLoads[i];
            return {
                time: ts.split(' ')[1]?.substring(0, 5) || ts,
                diff: (lv !== null && gv !== null && !isNaN(lv) && !isNaN(gv))
                    ? Math.round(Math.abs(lv - gv)) : 0
            };
        });

    // Fix 19: Find max divergence point for annotation
    const maxDivIdx   = varianceData.reduce((iMax, d, i) => d.diff > varianceData[iMax].diff ? i : iMax, 0);
    const maxDivPoint = varianceData[maxDivIdx];

    // Fix 21: Replace meaningless 50/50 cumulative pie with prediction-lead comparison
    const lstmLower = lstmLoads.filter((v, i) => v !== null && gruLoads[i] !== null && !isNaN(v) && !isNaN(gruLoads[i]) && v < gruLoads[i]).length;
    const gruLower  = lstmLoads.filter((v, i) => v !== null && gruLoads[i] !== null && !isNaN(v) && !isNaN(gruLoads[i]) && v > gruLoads[i]).length;
    const tied      = lstmLoads.length - lstmLower - gruLower;
    const modelLeadData = [
        { name: 'LSTM lower', value: lstmLower },
        { name: 'GRU lower',  value: gruLower  },
        { name: 'Tied',       value: tied       },
    ];

    const COLORS = ['#00FFF6', '#FF2A6D', '#F9F871'];
    const avgDiff = cleanLstm.reduce((acc, val, i) => acc + Math.abs(val - (cleanGru[i] ?? val)), 0) / (cleanLstm.length || 1);
    const deviationPct = lstmAvg ? ((avgDiff / lstmAvg) * 100).toFixed(2) : '0.00';

    return (
        <div className="space-y-5 font-mono w-full">
            <header className="border-b border-[#00FFF6]/30 pb-4">
                <h2 className="text-2xl font-bold text-[#00FFF6] flex items-center gap-3 uppercase">
                    <BarChart3 className="w-6 h-6" /> SYSTEM_ANALYTICS
                </h2>
                <p className="text-slate-500 text-xs mt-1">
                    {timestamps.length} predictions | {timestamps[0]} → {timestamps[timestamps.length - 1]}
                </p>
            </header>

            <div className="grid grid-cols-3 md:grid-cols-6 gap-2">
                <div className="bg-gradient-to-br from-[#0B0F1A] to-[#131b2d] border border-[#FF2A6D]/40 p-3 rounded">
                    <p className="text-[10px] text-slate-500 uppercase">Peak</p>
                    <p className="text-lg font-bold text-[#FF2A6D]">{Math.round(Math.max(lstmMax, gruMax)).toLocaleString()}</p>
                    <p className="text-[9px] text-slate-600">MW</p>
                </div>
                <div className="bg-gradient-to-br from-[#0B0F1A] to-[#131b2d] border border-[#00FFF6]/40 p-3 rounded">
                    <p className="text-[10px] text-slate-500 uppercase">Min</p>
                    <p className="text-lg font-bold text-[#00FFF6]">{Math.round(Math.min(lstmMin, gruMin)).toLocaleString()}</p>
                    <p className="text-[9px] text-slate-600">MW</p>
                </div>
                <div className="bg-gradient-to-br from-[#0B0F1A] to-[#131b2d] border border-white/20 p-3 rounded">
                    <p className="text-[10px] text-slate-500 uppercase">Avg</p>
                    <p className="text-lg font-bold text-white">{Math.round((lstmAvg + gruAvg) / 2).toLocaleString()}</p>
                    <p className="text-[9px] text-slate-600">MW</p>
                </div>
                <div className="bg-gradient-to-br from-[#0B0F1A] to-[#131b2d] border border-[#F9F871]/40 p-3 rounded">
                    <p className="text-[10px] text-slate-500 uppercase">Deviation</p>
                    <p className="text-lg font-bold text-[#F9F871]">{deviationPct}%</p>
                    <p className="text-[9px] text-slate-600">{Math.round(avgDiff)} MW</p>
                </div>
                <div className="bg-gradient-to-br from-[#0B0F1A] to-[#131b2d] border border-orange-500/40 p-3 rounded">
                    <p className="text-[10px] text-slate-500 uppercase">LSTM MAPE</p>
                    <p className="text-lg font-bold text-orange-400">{metrics?.lstm_hybrid?.mape_percent || 0}%</p>
                    <p className="text-[9px] text-slate-600">accuracy</p>
                </div>
                <div className="bg-gradient-to-br from-[#0B0F1A] to-[#131b2d] border border-purple-500/40 p-3 rounded">
                    <p className="text-[10px] text-slate-500 uppercase">GRU MAPE</p>
                    <p className="text-lg font-bold text-purple-400">{metrics?.gru_hybrid?.mape_percent || 0}%</p>
                    <p className="text-[9px] text-slate-600">accuracy</p>
                </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="bg-[#0B0F1A] border border-[#00FFF6]/30 p-4 rounded-lg">
                    <h3 className="text-sm font-bold text-[#00FFF6] mb-3 flex items-center gap-2">
                        <Cpu className="w-4 h-4" /> LSTM HYBRID
                    </h3>
                    <div className="grid grid-cols-3 gap-3 text-center">
                        <div className="bg-[#131b2d] p-3 rounded">
                            <ArrowUp className="w-4 h-4 text-[#FF2A6D] mx-auto mb-1" />
                            <p className="text-lg font-bold text-white">{Math.round(lstmMax).toLocaleString()}</p>
                            <p className="text-[9px] text-slate-500">PEAK</p>
                        </div>
                        <div className="bg-[#131b2d] p-3 rounded">
                            <ArrowDown className="w-4 h-4 text-[#00FFF6] mx-auto mb-1" />
                            <p className="text-lg font-bold text-white">{Math.round(lstmMin).toLocaleString()}</p>
                            <p className="text-[9px] text-slate-500">MIN</p>
                        </div>
                        <div className="bg-[#131b2d] p-3 rounded">
                            <Activity className="w-4 h-4 text-[#F9F871] mx-auto mb-1" />
                            <p className="text-lg font-bold text-white">{Math.round(lstmAvg).toLocaleString()}</p>
                            <p className="text-[9px] text-slate-500">AVG</p>
                        </div>
                    </div>
                </div>

                <div className="bg-[#0B0F1A] border border-[#FF2A6D]/30 p-4 rounded-lg">
                    <h3 className="text-sm font-bold text-[#FF2A6D] mb-3 flex items-center gap-2">
                        <Cpu className="w-4 h-4" /> GRU HYBRID
                    </h3>
                    <div className="grid grid-cols-3 gap-3 text-center">
                        <div className="bg-[#131b2d] p-3 rounded">
                            <ArrowUp className="w-4 h-4 text-[#FF2A6D] mx-auto mb-1" />
                            <p className="text-lg font-bold text-white">{Math.round(gruMax).toLocaleString()}</p>
                            <p className="text-[9px] text-slate-500">PEAK</p>
                        </div>
                        <div className="bg-[#131b2d] p-3 rounded">
                            <ArrowDown className="w-4 h-4 text-[#00FFF6] mx-auto mb-1" />
                            <p className="text-lg font-bold text-white">{Math.round(gruMin).toLocaleString()}</p>
                            <p className="text-[9px] text-slate-500">MIN</p>
                        </div>
                        <div className="bg-[#131b2d] p-3 rounded">
                            <Activity className="w-4 h-4 text-[#F9F871] mx-auto mb-1" />
                            <p className="text-lg font-bold text-white">{Math.round(gruAvg).toLocaleString()}</p>
                            <p className="text-[9px] text-slate-500">AVG</p>
                        </div>
                    </div>
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                <div className="bg-[#0B0F1A] border border-[#1e293b] p-4 rounded-lg">
                    <h3 className="text-sm font-bold text-slate-400 mb-2 flex items-center gap-2">
                        <Clock className="w-4 h-4 text-[#00FFF6]" /> HOURLY LOAD PATTERN
                    </h3>
                    <div style={{ width: '100%', height: 280 }}>
                        <ResponsiveContainer>
                            <AreaChart data={hourlyChartData} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
                                <defs>
                                    <linearGradient id="gL" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="5%" stopColor="#00FFF6" stopOpacity={0.4}/>
                                        <stop offset="95%" stopColor="#00FFF6" stopOpacity={0}/>
                                    </linearGradient>
                                    <linearGradient id="gG" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="5%" stopColor="#FF2A6D" stopOpacity={0.4}/>
                                        <stop offset="95%" stopColor="#FF2A6D" stopOpacity={0}/>
                                    </linearGradient>
                                </defs>
                                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                                <XAxis dataKey="hour" stroke="#64748b" tick={{ fontSize: 10, fill: '#64748b' }} />
                                <YAxis stroke="#64748b" tick={{ fontSize: 10, fill: '#64748b' }} tickFormatter={v => `${(v/1000).toFixed(1)}k`} domain={['dataMin - 200', 'dataMax + 200']} />
                                <Tooltip contentStyle={{ backgroundColor: '#0B0F1A', border: '1px solid #00FFF6', borderRadius: 4, fontSize: 11 }} />
                                <Legend wrapperStyle={{ fontSize: 11 }} />
                                {/* Fix 18: Peak annotation label on LSTM area */}
                                <Area type="monotone" dataKey="LSTM" stroke="#00FFF6" fill="url(#gL)" strokeWidth={2}
                                    label={(props) => {
                                        const peakHour = hourlyChartData.reduce((iMax, d, i) =>
                                            d.LSTM > hourlyChartData[iMax].LSTM ? i : iMax, 0);
                                        if (props.index !== peakHour) return null;
                                        return (
                                            <text x={props.x} y={props.y - 8} fill="#FF2A6D" fontSize={10}
                                                fontFamily="monospace" textAnchor="middle">
                                                PEAK: {props.value?.toLocaleString()} MW
                                            </text>
                                        );
                                    }}
                                />
                                <Area type="monotone" dataKey="GRU" stroke="#FF2A6D" fill="url(#gG)" strokeWidth={2} />
                            </AreaChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                <div className="bg-[#0B0F1A] border border-[#1e293b] p-4 rounded-lg">
                    <h3 className="text-sm font-bold text-slate-400 mb-2 flex items-center gap-2">
                        <BarChart3 className="w-4 h-4 text-[#FF2A6D]" /> MODEL COMPARISON
                    </h3>
                    <div style={{ width: '100%', height: 280 }}>
                        <ResponsiveContainer>
                            <BarChart data={comparisonData} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                                <XAxis dataKey="metric" stroke="#64748b" tick={{ fontSize: 11, fill: '#64748b' }} />
                                <YAxis stroke="#64748b" tick={{ fontSize: 10, fill: '#64748b' }} tickFormatter={v => `${(v/1000).toFixed(1)}k`} domain={[0, 'dataMax + 500']} />
                                <Tooltip contentStyle={{ backgroundColor: '#0B0F1A', border: '1px solid #FF2A6D', borderRadius: 4, fontSize: 11 }} formatter={(v) => [`${v.toLocaleString()} MW`]} />
                                <Legend wrapperStyle={{ fontSize: 11 }} />
                                {/* Fix 17: Show actual MW values as labels on each bar */}
                                <Bar dataKey="LSTM" fill="#00FFF6" radius={[4, 4, 0, 0]}>
                                    <LabelList content={(props) => {
                                        const { x, y, width, value } = props;
                                        return <text x={x + width / 2} y={y - 5} fill="#8A8F98"
                                            fontSize={10} fontFamily="monospace" textAnchor="middle">
                                            {value?.toLocaleString()}
                                        </text>;
                                    }} />
                                </Bar>
                                <Bar dataKey="GRU" fill="#FF2A6D" radius={[4, 4, 0, 0]}>
                                    <LabelList content={(props) => {
                                        const { x, y, width, value } = props;
                                        return <text x={x + width / 2} y={y - 5} fill="#8A8F98"
                                            fontSize={10} fontFamily="monospace" textAnchor="middle">
                                            {value?.toLocaleString()}
                                        </text>;
                                    }} />
                                </Bar>
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                <div className="bg-[#0B0F1A] border border-[#1e293b] p-4 rounded-lg">
                    <h3 className="text-sm font-bold text-slate-400 mb-2 flex items-center gap-2">
                        <Activity className="w-4 h-4 text-[#F9F871]" /> LOAD DISTRIBUTION
                    </h3>
                    <div style={{ width: '100%', height: 240 }}>
                        <ResponsiveContainer>
                            <BarChart data={distributionData} margin={{ top: 10, right: 10, left: -10, bottom: 30 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                                <XAxis dataKey="range" stroke="#64748b" tick={{ fontSize: 9, fill: '#64748b' }} angle={-45} textAnchor="end" />
                                <YAxis stroke="#64748b" tick={{ fontSize: 10, fill: '#64748b' }} />
                                <Tooltip contentStyle={{ backgroundColor: '#0B0F1A', border: '1px solid #F9F871', borderRadius: 4, fontSize: 11 }} />
                                <Bar dataKey="count" fill="#F9F871" radius={[4, 4, 0, 0]} name="Frequency" />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                <div className="bg-[#0B0F1A] border border-[#1e293b] p-4 rounded-lg">
                    <h3 className="text-sm font-bold text-slate-400 mb-2 flex items-center gap-2">
                        <TrendingUp className="w-4 h-4 text-purple-400" /> PREDICTION VARIANCE
                    </h3>
                    <div style={{ width: '100%', height: 240 }}>
                        <ResponsiveContainer>
                            <LineChart data={varianceData} margin={{ top: 20, right: 10, left: -10, bottom: 0 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                                <XAxis dataKey="time" stroke="#64748b" tick={{ fontSize: 9, fill: '#64748b' }} />
                                <YAxis stroke="#64748b" tick={{ fontSize: 10, fill: '#64748b' }} />
                                <Tooltip contentStyle={{ backgroundColor: '#0B0F1A', border: '1px solid #a855f7', borderRadius: 4, fontSize: 11 }} />
                                <Line type="monotone" dataKey="diff" stroke="#a855f7" strokeWidth={2} dot={false} name="Δ MW" />
                                {/* Fix 19: Max divergence annotation */}
                                {maxDivPoint && (
                                    <ReferenceDot
                                        x={maxDivPoint.time} y={maxDivPoint.diff}
                                        r={5} fill="#a855f7" stroke="#0B0F1A" strokeWidth={2}
                                        label={{ value: `Max Δ: ${maxDivPoint.diff} MW`, position: 'top', fill: '#a855f7', fontSize: 10, fontFamily: 'monospace' }}
                                    />
                                )}
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                {/* Fix 21: Replaced meaningless 50/50 cumulative split pie with prediction-lead pie */}
                <div className="bg-[#0B0F1A] border border-[#1e293b] p-4 rounded-lg">
                    <h3 className="text-sm font-bold text-slate-400 mb-2 flex items-center gap-2">
                        <Target className="w-4 h-4 text-green-400" /> PREDICTION LEAD
                    </h3>
                    <div style={{ width: '100%', height: 240 }}>
                        <ResponsiveContainer>
                            <PieChart>
                                <Pie data={modelLeadData} cx="50%" cy="50%" innerRadius={45} outerRadius={75} paddingAngle={5} dataKey="value">
                                    {modelLeadData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                                </Pie>
                                <Tooltip contentStyle={{ backgroundColor: '#0B0F1A', border: '1px solid #00FFF6', borderRadius: 4, fontSize: 11 }} formatter={(v, name) => [`${v.toLocaleString()} steps`, name]} />
                                <Legend wrapperStyle={{ fontSize: 11 }} />
                            </PieChart>
                        </ResponsiveContainer>
                    </div>
                </div>
            </div>

            {metrics && (
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="bg-gradient-to-r from-[#0B0F1A] to-[#131b2d] border border-[#00FFF6]/20 p-4 rounded-lg">
                        <h3 className="text-sm font-bold text-[#00FFF6] mb-3 flex items-center gap-2">
                            <Database className="w-4 h-4" /> TRAINING DATA
                        </h3>
                        <div className="space-y-2 text-xs">
                            <div className="flex justify-between"><span className="text-slate-500">Samples</span><span className="text-white">{metrics.lstm_hybrid.training_samples.toLocaleString()}</span></div>
                            <div className="flex justify-between"><span className="text-slate-500">Features</span><span className="text-white">{metrics.lstm_hybrid.features}</span></div>
                            <div className="flex justify-between"><span className="text-slate-500">Period</span><span className="text-white">{metrics.training_period}</span></div>
                        </div>
                    </div>
                    <div className="bg-gradient-to-r from-[#0B0F1A] to-[#131b2d] border border-[#00FFF6]/20 p-4 rounded-lg">
                        <h3 className="text-sm font-bold text-[#00FFF6] mb-3 flex items-center gap-2">
                            <Award className="w-4 h-4" /> LSTM METRICS
                        </h3>
                        <div className="space-y-2 text-xs">
                            <div className="flex justify-between"><span className="text-slate-500">RMSE</span><span className="text-[#00FFF6] font-bold">{metrics.lstm_hybrid.rmse_mw} MW</span></div>
                            <div className="flex justify-between"><span className="text-slate-500">MAPE</span><span className="text-[#00FFF6] font-bold">{metrics.lstm_hybrid.mape_percent}%</span></div>
                        </div>
                    </div>
                    <div className="bg-gradient-to-r from-[#0B0F1A] to-[#131b2d] border border-[#FF2A6D]/20 p-4 rounded-lg">
                        <h3 className="text-sm font-bold text-[#FF2A6D] mb-3 flex items-center gap-2">
                            <Award className="w-4 h-4" /> GRU METRICS
                        </h3>
                        <div className="space-y-2 text-xs">
                            <div className="flex justify-between"><span className="text-slate-500">RMSE</span><span className="text-[#FF2A6D] font-bold">{metrics.gru_hybrid.rmse_mw} MW</span></div>
                            <div className="flex justify-between"><span className="text-slate-500">MAPE</span><span className="text-[#FF2A6D] font-bold">{metrics.gru_hybrid.mape_percent}%</span></div>
                        </div>
                    </div>
                </div>
            )}

            <div className="bg-[#0B0F1A]/50 border border-[#1e293b] p-3 rounded flex flex-wrap justify-between items-center text-xs text-slate-500">
                <div className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                    Analysis Complete
                </div>
                <div>Range: <span className="text-[#00FFF6]">{Math.round(Math.min(lstmMin, gruMin)).toLocaleString()}</span> — <span className="text-[#FF2A6D]">{Math.round(Math.max(lstmMax, gruMax)).toLocaleString()} MW</span></div>
                <div>Points: <span className="text-white">{timestamps.length.toLocaleString()}</span></div>
            </div>
        </div>
    );
};

export default Analytics;
