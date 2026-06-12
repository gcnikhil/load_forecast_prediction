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
                    if (!result || !result.loads_lightgbm_gru) {
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

    const hasLstm = lstmLoads && lstmLoads.length > 0;
    const cleanLstm = hasLstm ? lstmLoads.filter(v => v !== null && v !== undefined && !isNaN(v)) : [];
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
        if (hasLstm) {
            hourlyMap[hour].lstm.push(lstmLoads[i]);
        }
        hourlyMap[hour].gru.push(gruLoads[i]);
    });

    const hourlyChartData = Object.keys(hourlyMap)
        .sort((a, b) => parseInt(a) - parseInt(b))
        .map(hour => {
            const row = {
                hour: `${hour}:00`,
                GRU: Math.round(hourlyMap[hour].gru.reduce((a, b) => a + b, 0) / hourlyMap[hour].gru.length)
            };
            if (hasLstm) {
                row.LSTM = Math.round(hourlyMap[hour].lstm.reduce((a, b) => a + b, 0) / hourlyMap[hour].lstm.length);
            }
            return row;
        });

    const comparisonData = hasLstm ? [
        { metric: 'Peak', LSTM: Math.round(lstmMax), GRU: Math.round(gruMax) },
        { metric: 'Min', LSTM: Math.round(lstmMin), GRU: Math.round(gruMin) },
        { metric: 'Average', LSTM: Math.round(lstmAvg), GRU: Math.round(gruAvg) }
    ] : [
        { metric: 'Peak', GRU: Math.round(gruMax) },
        { metric: 'Min', GRU: Math.round(gruMin) },
        { metric: 'Average', GRU: Math.round(gruAvg) }
    ];

    const distributionMap = {};
    const allValidLoads = hasLstm ? [...lstmLoads, ...gruLoads] : [...gruLoads];
    allValidLoads.forEach(load => {
        if (load === null || load === undefined || isNaN(load)) return;
        const bucket = Math.floor(load / 500) * 500;
        const key = `${(bucket/1000).toFixed(1)}-${((bucket+500)/1000).toFixed(1)}k`;
        distributionMap[key] = (distributionMap[key] || 0) + 1;
    });
    const distributionData = Object.entries(distributionMap)
        .map(([range, count]) => ({ range, count }))
        .sort((a, b) => parseFloat(a.range) - parseFloat(b.range));

    const varianceData = hasLstm ? timestamps
        .filter((_, i) => i % 30 === 0)
        .map((ts, idx) => {
            const i = idx * 30;
            const lv = lstmLoads[i]; const gv = gruLoads[i];
            return {
                time: ts.split(' ')[1]?.substring(0, 5) || ts,
                diff: (lv !== null && gv !== null && !isNaN(lv) && !isNaN(gv))
                    ? Math.round(Math.abs(lv - gv)) : 0
            };
        }) : [];

    // Find max divergence point for annotation if variance exists
    const maxDivIdx   = varianceData.length ? varianceData.reduce((iMax, d, i) => d.diff > varianceData[iMax].diff ? i : iMax, 0) : 0;
    const maxDivPoint = varianceData.length ? varianceData[maxDivIdx] : null;

    // Build pie data
    let pieData = [];
    let COLORS = [];
    if (hasLstm) {
        const lstmLower = lstmLoads.filter((v, i) => v !== null && gruLoads[i] !== null && !isNaN(v) && !isNaN(gruLoads[i]) && v < gruLoads[i]).length;
        const gruLower  = lstmLoads.filter((v, i) => v !== null && gruLoads[i] !== null && !isNaN(v) && !isNaN(gruLoads[i]) && v > gruLoads[i]).length;
        const tied      = lstmLoads.length - lstmLower - gruLower;
        pieData = [
            { name: 'LSTM lower', value: lstmLower },
            { name: 'GRU lower',  value: gruLower  },
            { name: 'Tied',       value: tied       },
        ];
        COLORS = ['#00FFF6', '#FF2A6D', '#F9F871'];
    } else {
        // Compare daytime vs nighttime load for GRU predictions
        let dayLoads = 0, nightLoads = 0;
        timestamps.forEach((ts, i) => {
            const hour = parseInt(ts.split(' ')[1]?.split(':')[0] || '00');
            const val = gruLoads[i];
            if (val === null || val === undefined || isNaN(val)) return;
            if (hour >= 6 && hour < 18) {
                dayLoads += val;
            } else {
                nightLoads += val;
            }
        });
        pieData = [
            { name: 'Daytime (6am-6pm)', value: Math.round(dayLoads / 1000) },
            { name: 'Nighttime (6pm-6am)', value: Math.round(nightLoads / 1000) }
        ];
        COLORS = ['#F9F871', '#FF2A6D'];
    }

    const avgDiff = hasLstm ? cleanLstm.reduce((acc, val, i) => acc + Math.abs(val - (cleanGru[i] ?? val)), 0) / (cleanLstm.length || 1) : 0;
    const deviationPct = (hasLstm && lstmAvg) ? ((avgDiff / lstmAvg) * 100).toFixed(2) : '0.00';

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

            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-2">
                <div className="bg-gradient-to-br from-[#0B0F1A] to-[#131b2d] border border-[#FF2A6D]/40 p-3 rounded">
                    <p className="text-[10px] text-slate-500 uppercase">Peak Load</p>
                    <p className="text-lg font-bold text-[#FF2A6D]">
                        {Math.round(hasLstm ? Math.max(lstmMax, gruMax) : gruMax).toLocaleString()}
                    </p>
                    <p className="text-[9px] text-slate-600">MW</p>
                </div>
                <div className="bg-gradient-to-br from-[#0B0F1A] to-[#131b2d] border border-[#00FFF6]/40 p-3 rounded">
                    <p className="text-[10px] text-slate-500 uppercase">Min Load</p>
                    <p className="text-lg font-bold text-[#00FFF6]">
                        {Math.round(hasLstm ? Math.min(lstmMin, gruMin) : gruMin).toLocaleString()}
                    </p>
                    <p className="text-[9px] text-slate-600">MW</p>
                </div>
                <div className="bg-gradient-to-br from-[#0B0F1A] to-[#131b2d] border border-white/20 p-3 rounded">
                    <p className="text-[10px] text-slate-500 uppercase">Avg Load</p>
                    <p className="text-lg font-bold text-white">
                        {Math.round(hasLstm ? (lstmAvg + gruAvg) / 2 : gruAvg).toLocaleString()}
                    </p>
                    <p className="text-[9px] text-slate-600">MW</p>
                </div>
                {hasLstm ? (
                    <div className="bg-gradient-to-br from-[#0B0F1A] to-[#131b2d] border border-[#F9F871]/40 p-3 rounded">
                        <p className="text-[10px] text-slate-500 uppercase">Deviation</p>
                        <p className="text-lg font-bold text-[#F9F871]">{deviationPct}%</p>
                        <p className="text-[9px] text-slate-600">{Math.round(avgDiff)} MW</p>
                    </div>
                ) : (
                    <div className="bg-gradient-to-br from-[#0B0F1A] to-[#131b2d] border border-green-500/40 p-3 rounded">
                        <p className="text-[10px] text-slate-500 uppercase">Active Model</p>
                        <p className="text-lg font-bold text-green-400">GRU_HYBRID</p>
                        <p className="text-[9px] text-slate-600">running</p>
                    </div>
                )}
                {hasLstm && (
                    <div className="bg-gradient-to-br from-[#0B0F1A] to-[#131b2d] border border-orange-500/40 p-3 rounded">
                        <p className="text-[10px] text-slate-500 uppercase">LSTM MAPE</p>
                        <p className="text-lg font-bold text-orange-400">{metrics?.lstm_hybrid?.mape_percent || 0}%</p>
                        <p className="text-[9px] text-slate-600">accuracy</p>
                    </div>
                )}
                <div className="bg-gradient-to-br from-[#0B0F1A] to-[#131b2d] border border-purple-500/40 p-3 rounded col-span-2 md:col-span-1">
                    <p className="text-[10px] text-slate-500 uppercase">GRU MAPE</p>
                    <p className="text-lg font-bold text-purple-400">{metrics?.gru_hybrid?.mape_percent || 0}%</p>
                    <p className="text-[9px] text-slate-600">accuracy</p>
                </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {hasLstm && (
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
                )}

                <div className={`bg-[#0B0F1A] border border-[#FF2A6D]/30 p-4 rounded-lg ${!hasLstm ? 'col-span-2' : ''}`}>
                    <h3 className="text-sm font-bold text-[#FF2A6D] mb-3 flex items-center gap-2">
                        <Cpu className="w-4 h-4" /> GRU HYBRID MODEL ANALYSIS
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
                                {hasLstm && (
                                    <Area type="monotone" dataKey="LSTM" stroke="#00FFF6" fill="url(#gL)" strokeWidth={2}
                                        label={(props) => {
                                            const peakHour = hourlyChartData.reduce((iMax, d, i) =>
                                                d.LSTM > hourlyChartData[iMax].LSTM ? i : iMax, 0);
                                            if (props.index !== peakHour) return null;
                                            return (
                                                <text x={props.x} y={props.y - 8} fill="#00FFF6" fontSize={10}
                                                    fontFamily="monospace" textAnchor="middle">
                                                    PEAK: {props.value?.toLocaleString()} MW
                                                </text>
                                            );
                                        }}
                                    />
                                )}
                                <Area type="monotone" dataKey="GRU" stroke="#FF2A6D" fill="url(#gG)" strokeWidth={2} />
                            </AreaChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                <div className="bg-[#0B0F1A] border border-[#1e293b] p-4 rounded-lg">
                    <h3 className="text-sm font-bold text-slate-400 mb-2 flex items-center gap-2">
                        <BarChart3 className="w-4 h-4 text-[#FF2A6D]" /> FORECAST COMPARISON BY METRICS
                    </h3>
                    <div style={{ width: '100%', height: 280 }}>
                        <ResponsiveContainer>
                            <BarChart data={comparisonData} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                                <XAxis dataKey="metric" stroke="#64748b" tick={{ fontSize: 11, fill: '#64748b' }} />
                                <YAxis stroke="#64748b" tick={{ fontSize: 10, fill: '#64748b' }} tickFormatter={v => `${(v/1000).toFixed(1)}k`} domain={[0, 'dataMax + 500']} />
                                <Tooltip contentStyle={{ backgroundColor: '#0B0F1A', border: '1px solid #FF2A6D', borderRadius: 4, fontSize: 11 }} formatter={(v) => [`${v.toLocaleString()} MW`]} />
                                <Legend wrapperStyle={{ fontSize: 11 }} />
                                {hasLstm && (
                                    <Bar dataKey="LSTM" fill="#00FFF6" radius={[4, 4, 0, 0]}>
                                        <LabelList content={(props) => {
                                            const { x, y, width, value } = props;
                                            return <text x={x + width / 2} y={y - 5} fill="#8A8F98"
                                                fontSize={10} fontFamily="monospace" textAnchor="middle">
                                                {value?.toLocaleString()}
                                            </text>;
                                        }} />
                                    </Bar>
                                )}
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
                            {hasLstm ? (
                                <LineChart data={varianceData} margin={{ top: 20, right: 10, left: -10, bottom: 0 }}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                                    <XAxis dataKey="time" stroke="#64748b" tick={{ fontSize: 9, fill: '#64748b' }} />
                                    <YAxis stroke="#64748b" tick={{ fontSize: 10, fill: '#64748b' }} />
                                    <Tooltip contentStyle={{ backgroundColor: '#0B0F1A', border: '1px solid #a855f7', borderRadius: 4, fontSize: 11 }} />
                                    <Line type="monotone" dataKey="diff" stroke="#a855f7" strokeWidth={2} dot={false} name="Δ MW" />
                                    {maxDivPoint && (
                                        <ReferenceDot
                                            x={maxDivPoint.time} y={maxDivPoint.diff}
                                            r={5} fill="#a855f7" stroke="#0B0F1A" strokeWidth={2}
                                            label={{ value: `Max Δ: ${maxDivPoint.diff} MW`, position: 'top', fill: '#a855f7', fontSize: 10, fontFamily: 'monospace' }}
                                        />
                                    )}
                                </LineChart>
                            ) : (
                                <div className="h-full flex flex-col items-center justify-center text-[#8A8F98] space-y-2 text-xs text-center p-4">
                                    <TrendingUp className="w-8 h-8 text-slate-600 mb-2" />
                                    <p>SINGLE MODEL ACTIVE</p>
                                    <p className="text-[10px] text-slate-500">Divergence analysis requires multiple models to compare.</p>
                                </div>
                            )}
                        </ResponsiveContainer>
                    </div>
                </div>

                <div className="bg-[#0B0F1A] border border-[#1e293b] p-4 rounded-lg">
                    <h3 className="text-sm font-bold text-slate-400 mb-2 flex items-center gap-2">
                        <Target className="w-4 h-4 text-green-400" /> {hasLstm ? 'PREDICTION LEAD' : 'DEMAND PROFILE SHARE'}
                    </h3>
                    <div style={{ width: '100%', height: 240 }}>
                        <ResponsiveContainer>
                            <PieChart>
                                <Pie data={pieData} cx="50%" cy="50%" innerRadius={45} outerRadius={75} paddingAngle={5} dataKey="value">
                                    {pieData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                                </Pie>
                                {hasLstm ? (
                                    <Tooltip contentStyle={{ backgroundColor: '#0B0F1A', border: '1px solid #00FFF6', borderRadius: 4, fontSize: 11 }} formatter={(v, name) => [`${v.toLocaleString()} steps`, name]} />
                                ) : (
                                    <Tooltip contentStyle={{ backgroundColor: '#0B0F1A', border: '1px solid #00FFF6', borderRadius: 4, fontSize: 11 }} formatter={(v, name) => [`${v.toLocaleString()} GWh`, name]} />
                                )}
                                <Legend wrapperStyle={{ fontSize: 11 }} />
                            </PieChart>
                        </ResponsiveContainer>
                    </div>
                </div>
            </div>

            {metrics && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="bg-gradient-to-r from-[#0B0F1A] to-[#131b2d] border border-[#00FFF6]/20 p-4 rounded-lg">
                        <h3 className="text-sm font-bold text-[#00FFF6] mb-3 flex items-center gap-2">
                            <Database className="w-4 h-4" /> TRAINING DATA
                        </h3>
                        <div className="space-y-2 text-xs">
                            <div className="flex justify-between"><span className="text-slate-500">Samples</span><span className="text-white">{(metrics.gru_hybrid.training_samples || metrics.lstm_hybrid?.training_samples || 0).toLocaleString()}</span></div>
                            <div className="flex justify-between"><span className="text-slate-500">Features</span><span className="text-white">{metrics.gru_hybrid.features || metrics.lstm_hybrid?.features}</span></div>
                            <div className="flex justify-between"><span className="text-slate-500">Period</span><span className="text-white">{metrics.training_period}</span></div>
                        </div>
                    </div>
                    <div className="bg-gradient-to-r from-[#0B0F1A] to-[#131b2d] border border-[#FF2A6D]/20 p-4 rounded-lg">
                        <h3 className="text-sm font-bold text-[#FF2A6D] mb-3 flex items-center gap-2">
                            <Award className="w-4 h-4" /> GRU MODEL PERFORMANCE
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
                <div>Range: <span className="text-[#00FFF6]">{Math.round(gruMin).toLocaleString()}</span> — <span className="text-[#FF2A6D]">{Math.round(gruMax).toLocaleString()} MW</span></div>
                <div>Points: <span className="text-white">{timestamps.length.toLocaleString()}</span></div>
            </div>
        </div>
    );
};

export default Analytics;
