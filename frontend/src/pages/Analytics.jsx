import { useState, useEffect } from 'react';
import { useData } from '../contexts/DataContext';
import {
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
    LineChart, Line, AreaChart, Area, PieChart, Pie, Cell
} from 'recharts';
import {
    predictLoad, getModelMetrics, getHistoricalAccuracy,
    getFeatureImportance, getForecastVsActual
} from '../services/api';
import {
    TrendingUp, Activity, AlertTriangle, Target, Award, Clock,
    BarChart3, Cpu, Database, ArrowUp, ArrowDown, Download, FileText
} from 'lucide-react';

const Analytics = () => {
    const { data: sharedData, setData: setSharedData, metrics: sharedMetrics, setMetrics: setSharedMetrics } = useData();
    const [data, setData] = useState(sharedData);
    const [loading, setLoading] = useState(!sharedData);
    const [error, setError] = useState(null);
    const [metrics, setMetrics] = useState(sharedMetrics);
    const [accuracy, setAccuracy] = useState(null);
    const [featureImportance, setFeatureImportance] = useState([]);
    const [forecastVsActual, setForecastVsActual] = useState(null);
    const [loadingExtra, setLoadingExtra] = useState(true);

    useEffect(() => {
        const fetchExtraData = async () => {
            try {
                setLoadingExtra(true);
                const [featImp, fva] = await Promise.all([
                    getFeatureImportance().catch(err => {
                        console.error('Feature importance fetch failed:', err);
                        return null;
                    }),
                    getForecastVsActual(7).catch(err => {
                        console.error('Forecast vs Actual fetch failed:', err);
                        return null;
                    })
                ]);
                if (featImp?.features) {
                    setFeatureImportance(featImp.features.slice(0, 15));
                }
                if (fva) {
                    setForecastVsActual(fva);
                }
            } catch (err) {
                console.error("Extra analytics data fetch failed", err);
            } finally {
                setLoadingExtra(false);
            }
        };
        fetchExtraData();
    }, []);

    useEffect(() => {
        if (sharedData) {
            setData(sharedData);
            setLoading(false);
        }
        
        if (sharedMetrics) {
            setMetrics(sharedMetrics);
            if (sharedData) {
                // If sharedData and sharedMetrics are already there, fetch accuracy if not fetched
                const fetchAccOnly = async () => {
                    try {
                        const acc = await getHistoricalAccuracy();
                        setAccuracy(acc);
                    } catch (e) {
                        console.error('Historical accuracy fetch failed:', e);
                    }
                };
                fetchAccOnly();
                return;
            }
        }

        const fetchMetricsAndAccuracy = async () => {
            try {
                const [metricsData, accuracyData] = await Promise.all([
                    getModelMetrics(),
                    getHistoricalAccuracy().catch(err => {
                        console.error('Accuracy fetch failed:', err);
                        return null;
                    })
                ]);
                setMetrics(metricsData);
                setSharedMetrics(metricsData);
                if (accuracyData) setAccuracy(accuracyData);
            } catch (err) {
                console.error('Metrics fetch failed:', err);
            }
        };

        const fetchAll = async () => {
            try {
                const today = new Date();
                const yesterday = new Date(today);
                yesterday.setDate(today.getDate() - 1);
                const formatDate = (d) => d.toISOString().split('T')[0];
                const [result, metricsData, accuracyData] = await Promise.all([
                    predictLoad(formatDate(yesterday), formatDate(today)),
                    getModelMetrics(),
                    getHistoricalAccuracy().catch(err => {
                        console.error('Accuracy fetch failed:', err);
                        return null;
                    })
                ]);
                if (!result || !result.loads_lightgbm_gru) {
                    throw new Error('Invalid data format');
                }
                setData(result);
                setMetrics(metricsData);
                setSharedData(result);
                setSharedMetrics(metricsData);
                if (accuracyData) setAccuracy(accuracyData);
            } catch (err) {
                console.error('Analytics error:', err);
                setError(err.message);
            } finally {
                setLoading(false);
            }
        };

        if (!sharedData) {
            fetchAll();
        } else {
            fetchMetricsAndAccuracy();
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

    const gruLoads = data.loads_lightgbm_gru;
    const timestamps = data.timestamps;

    const cleanGru  = gruLoads.filter(v => v !== null && v !== undefined && !isNaN(v));
    
    const gruMax  = cleanGru.length  ? Math.max(...cleanGru)  : 0;
    const gruMin  = cleanGru.length  ? Math.min(...cleanGru)  : 0;
    const gruAvg  = cleanGru.length  ? cleanGru.reduce((a, b) => a + b, 0) / cleanGru.length : 0;

    const maxIdx = cleanGru.length ? gruLoads.indexOf(gruMax) : -1;
    const minIdx = cleanGru.length ? gruLoads.indexOf(gruMin) : -1;
    const peakTime = maxIdx !== -1 ? timestamps[maxIdx] : 'N/A';
    const minTime = minIdx !== -1 ? timestamps[minIdx] : 'N/A';

    const formatTime = (tsStr) => {
        if (!tsStr || tsStr === 'N/A') return 'N/A';
        const parts = tsStr.split(' ');
        if (parts.length > 1) {
            const dateParts = parts[0].split('-');
            const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
            const monthName = months[parseInt(dateParts[1]) - 1] || dateParts[1];
            return `${monthName} ${dateParts[2]} ${parts[1]}`;
        }
        return tsStr;
    };

    const loadFactor = gruMax > 0 ? ((gruAvg / gruMax) * 100).toFixed(1) : '0.0';

    const hourlyMap = {};
    timestamps.forEach((ts, i) => {
        const hour = ts.split(' ')[1]?.split(':')[0] || '00';
        if (!hourlyMap[hour]) hourlyMap[hour] = { gru: [] };
        hourlyMap[hour].gru.push(gruLoads[i]);
    });

    const hourlyChartData = Object.keys(hourlyMap)
        .sort((a, b) => parseInt(a) - parseInt(b))
        .map(hour => ({
            hour: `${hour}:00`,
            GRU: Math.round(hourlyMap[hour].gru.reduce((a, b) => a + b, 0) / hourlyMap[hour].gru.length)
        }));

    // Generate Load Duration Curve (LDC) data
    const sortedLoads = [...cleanGru].sort((a, b) => b - a);
    const ldcData = sortedLoads.map((load, index) => ({
        percentage: Math.round(((index + 1) / sortedLoads.length) * 100),
        load: Math.round(load)
    }));
    // Downsample for rendering performance
    const step = Math.max(1, Math.floor(ldcData.length / 50));
    const downsampledLdcData = ldcData.filter((_, idx) => idx % step === 0);

    const distributionMap = {};
    cleanGru.forEach(load => {
        const bucket = Math.floor(load / 500) * 500;
        const key = `${(bucket/1000).toFixed(1)}-${((bucket+500)/1000).toFixed(1)}k`;
        distributionMap[key] = (distributionMap[key] || 0) + 1;
    });
    const distributionData = Object.entries(distributionMap)
        .map(([range, count]) => ({ range, count }))
        .sort((a, b) => parseFloat(a.range) - parseFloat(b.range));

    // Compare morning vs daytime vs evening vs night loads
    let morningPeak = 0, morningCount = 0;
    let daytimeOffPeak = 0, daytimeCount = 0;
    let eveningPeak = 0, eveningCount = 0;
    let nighttimeOffPeak = 0, nighttimeCount = 0;

    timestamps.forEach((ts, i) => {
        const hour = parseInt(ts.split(' ')[1]?.split(':')[0] || '00');
        const val = gruLoads[i];
        if (val === null || val === undefined || isNaN(val)) return;

        if (hour >= 6 && hour < 10) {
            morningPeak += val;
            morningCount++;
        } else if (hour >= 10 && hour < 18) {
            daytimeOffPeak += val;
            daytimeCount++;
        } else if (hour >= 18 && hour < 22) {
            eveningPeak += val;
            eveningCount++;
        } else {
            nighttimeOffPeak += val;
            nighttimeCount++;
        }
    });

    const pieData = [
        { name: 'Morning Peak (6am-10am)', value: morningCount ? Math.round(morningPeak / morningCount) : 0 },
        { name: 'Day Off-Peak (10am-6pm)', value: daytimeCount ? Math.round(daytimeOffPeak / daytimeCount) : 0 },
        { name: 'Evening Peak (6pm-10pm)', value: eveningCount ? Math.round(eveningPeak / eveningCount) : 0 },
        { name: 'Night Off-Peak (10pm-6am)', value: nighttimeCount ? Math.round(nighttimeOffPeak / nighttimeCount) : 0 }
    ].filter(item => item.value > 0);

    const COLORS = ['#FF2A6D', '#00FFF6', '#F9F871', '#a855f7'];

    const accuracyChartData = accuracy?.daily_accuracy ? accuracy.daily_accuracy.map(d => ({
        date: d.date.substring(5), // "MM-DD"
        MAPE: d.gru_mape,
        RMSE: d.gru_rmse
    })) : [];

    const downloadCSV = () => {
        if (!data || !data.timestamps) return;
        const rows = [["Timestamp", "Predicted Load (MW)"]];
        data.timestamps.forEach((ts, idx) => {
            const loadVal = data.loads_lightgbm_gru[idx] !== null && data.loads_lightgbm_gru[idx] !== undefined
                ? data.loads_lightgbm_gru[idx].toFixed(1)
                : "N/A";
            rows.push([ts, loadVal]);
        });
        const csvContent = rows.map(e => e.map(val => `"${val}"`).join(",")).join("\n");
        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.setAttribute("href", url);
        const start = data.timestamps[0]?.split(' ')[0] || "start";
        const end = data.timestamps[data.timestamps.length - 1]?.split(' ')[0] || "end";
        link.setAttribute("download", `bescom_forecast_report_${start}_to_${end}.csv`);
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
    };

    const downloadPDF = () => {
        window.print();
    };

    const CustomOverlayTooltip = ({ active, payload, label }) => {
        if (active && payload && payload.length) {
            const act = payload.find(p => p.name === 'Actual')?.value;
            const pred = payload.find(p => p.name === 'Predicted')?.value;
            let diff = null;
            let pct = null;
            if (act !== null && pred !== null && act !== undefined && pred !== undefined) {
                diff = act - pred;
                pct = ((diff / act) * 100).toFixed(2);
            }
            return (
                <div className="bg-[#0B0F1A] border border-[#00FFF6] p-3 rounded text-xs font-mono">
                    <p className="text-slate-400 mb-1 font-bold">{label}</p>
                    {act !== null && act !== undefined && (
                        <p className="text-[#00FFF6]">ACTUAL: {act.toLocaleString()} MW</p>
                    )}
                    {pred !== null && pred !== undefined && (
                        <p className="text-[#FF2A6D]">PREDICTED: {pred.toLocaleString()} MW</p>
                    )}
                    {diff !== null && (
                        <p className={diff >= 0 ? "text-green-400" : "text-red-400"}>
                            ERROR: {diff > 0 ? `+${diff.toLocaleString()}` : diff.toLocaleString()} MW ({diff > 0 ? `+${pct}` : pct}%)
                        </p>
                    )}
                </div>
            );
        }
        return null;
    };

    let overlayChartData = [];
    if (forecastVsActual) {
        const step = Math.max(1, Math.floor(forecastVsActual.timestamps.length / 168));
        overlayChartData = forecastVsActual.timestamps
            .map((ts, idx) => ({
                timestamp: ts.substring(5),
                Actual: forecastVsActual.actual[idx] !== null ? Math.round(forecastVsActual.actual[idx]) : null,
                Predicted: forecastVsActual.predicted[idx] !== null ? Math.round(forecastVsActual.predicted[idx]) : null,
            }))
            .filter((_, idx) => idx % step === 0);
    }

    return (
        <div className="space-y-5 font-mono w-full">
            <header className="border-b border-[#00FFF6]/30 pb-4 flex flex-col sm:flex-row sm:justify-between sm:items-center gap-3">
                <div>
                    <h2 className="text-2xl font-bold text-[#00FFF6] flex items-center gap-3 uppercase">
                        <BarChart3 className="w-6 h-6" /> SYSTEM_ANALYTICS
                    </h2>
                    <p className="text-slate-500 text-xs mt-1">
                        {timestamps.length} predictions | {timestamps[0]} → {timestamps[timestamps.length - 1]}
                    </p>
                </div>
                <div className="flex gap-2 print-hide">
                    <button
                        onClick={downloadCSV}
                        className="border border-[#00FFF6] text-[#00FFF6] hover:bg-[#00FFF6] hover:text-[#0B0F1A] transition px-3 py-1.5 rounded text-xs uppercase flex items-center gap-1.5 font-bold cursor-pointer"
                        title="Download raw forecast data as CSV"
                    >
                        <Download className="w-3.5 h-3.5" /> CSV
                    </button>
                    <button
                        onClick={downloadPDF}
                        className="border border-[#FF2A6D] text-[#FF2A6D] hover:bg-[#FF2A6D] hover:text-[#0B0F1A] transition px-3 py-1.5 rounded text-xs uppercase flex items-center gap-1.5 font-bold cursor-pointer"
                        title="Print or Save Report as PDF"
                    >
                        <FileText className="w-3.5 h-3.5" /> PDF REPORT
                    </button>
                </div>
            </header>

            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-8 gap-2">
                <div className="bg-gradient-to-br from-[#0B0F1A] to-[#131b2d] border border-[#FF2A6D]/40 p-3 rounded">
                    <p className="text-[10px] text-slate-500 uppercase">Peak Load</p>
                    <p className="text-lg font-bold text-[#FF2A6D]">{Math.round(gruMax).toLocaleString()}</p>
                    <p className="text-[9px] text-slate-600">MW</p>
                </div>
                <div className="bg-gradient-to-br from-[#0B0F1A] to-[#131b2d] border border-[#FF2A6D]/20 p-3 rounded">
                    <p className="text-[10px] text-slate-500 uppercase">Peak Time</p>
                    <p className="text-[11px] font-bold text-[#FF2A6D]/80 mt-1 whitespace-nowrap overflow-hidden text-ellipsis">{formatTime(peakTime)}</p>
                </div>
                <div className="bg-gradient-to-br from-[#0B0F1A] to-[#131b2d] border border-[#00FFF6]/40 p-3 rounded">
                    <p className="text-[10px] text-slate-500 uppercase">Min Load</p>
                    <p className="text-lg font-bold text-[#00FFF6]">{Math.round(gruMin).toLocaleString()}</p>
                    <p className="text-[9px] text-slate-600">MW</p>
                </div>
                <div className="bg-gradient-to-br from-[#0B0F1A] to-[#131b2d] border border-[#00FFF6]/20 p-3 rounded">
                    <p className="text-[10px] text-slate-500 uppercase">Min Time</p>
                    <p className="text-[11px] font-bold text-[#00FFF6]/80 mt-1 whitespace-nowrap overflow-hidden text-ellipsis">{formatTime(minTime)}</p>
                </div>
                <div className="bg-gradient-to-br from-[#0B0F1A] to-[#131b2d] border border-white/20 p-3 rounded">
                    <p className="text-[10px] text-slate-500 uppercase">Avg Load</p>
                    <p className="text-lg font-bold text-white">{Math.round(gruAvg).toLocaleString()}</p>
                    <p className="text-[9px] text-slate-600">MW</p>
                </div>
                <div className="bg-gradient-to-br from-[#0B0F1A] to-[#131b2d] border border-[#F9F871]/40 p-3 rounded">
                    <p className="text-[10px] text-slate-500 uppercase">Load Factor</p>
                    <p className="text-lg font-bold text-[#F9F871]">{loadFactor}%</p>
                    <p className="text-[9px] text-slate-600">Avg/Peak Ratio</p>
                </div>
                <div className="bg-gradient-to-br from-[#0B0F1A] to-[#131b2d] border border-green-500/40 p-3 rounded">
                    <p className="text-[10px] text-slate-500 uppercase">Active Model</p>
                    <p className="text-lg font-bold text-green-400">GRU_HYBRID</p>
                    <p className="text-[9px] text-slate-600">running</p>
                </div>
                <div className="bg-gradient-to-br from-[#0B0F1A] to-[#131b2d] border border-purple-500/40 p-3 rounded">
                    <p className="text-[10px] text-slate-500 uppercase">Model MAPE</p>
                    <p className="text-lg font-bold text-purple-400">{metrics?.gru_hybrid?.mape_percent || 0}%</p>
                    <p className="text-[9px] text-slate-600">accuracy</p>
                </div>
            </div>

            <div className="bg-[#0B0F1A] border border-[#FF2A6D]/30 p-4 rounded-lg">
                <h3 className="text-sm font-bold text-[#FF2A6D] mb-3 flex items-center gap-2">
                    <Cpu className="w-4 h-4" /> GRU HYBRID MODEL FORECAST ANALYSIS
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4 text-center">
                    <div className="bg-[#131b2d] p-3 rounded">
                        <ArrowUp className="w-4 h-4 text-[#FF2A6D] mx-auto mb-1" />
                        <p className="text-lg font-bold text-white">{Math.round(gruMax).toLocaleString()} MW</p>
                        <p className="text-[9px] text-slate-500">PEAK FORECAST</p>
                    </div>
                    <div className="bg-[#131b2d] p-3 rounded">
                        <ArrowDown className="w-4 h-4 text-[#00FFF6] mx-auto mb-1" />
                        <p className="text-lg font-bold text-white">{Math.round(gruMin).toLocaleString()} MW</p>
                        <p className="text-[9px] text-slate-500">MIN FORECAST</p>
                    </div>
                    <div className="bg-[#131b2d] p-3 rounded">
                        <Activity className="w-4 h-4 text-[#F9F871] mx-auto mb-1" />
                        <p className="text-lg font-bold text-white">{Math.round(gruAvg).toLocaleString()} MW</p>
                        <p className="text-[9px] text-slate-500">AVERAGE FORECAST</p>
                    </div>
                    <div className="bg-[#131b2d] p-3 rounded flex flex-col justify-center">
                        <p className="text-xs text-[#00FFF6] uppercase font-bold">Grid Stability</p>
                        <p className="text-base text-slate-300 font-bold mt-1">
                            {parseFloat(loadFactor) > 80 ? 'HIGH (Optimal)' : parseFloat(loadFactor) > 60 ? 'MODERATE' : 'LOW (Volatile)'}
                        </p>
                        <p className="text-[9px] text-slate-500 mt-1">Based on Load Factor</p>
                    </div>
                </div>
            </div>

            {/* Forecast vs Actual Overlay Chart */}
            <div className="bg-[#0B0F1A] border border-[#00FFF6]/30 p-4 rounded-lg">
                <h3 className="text-sm font-bold text-[#00FFF6] mb-3 flex items-center gap-2">
                    <Activity className="w-4 h-4 text-[#00FFF6]" /> FORECAST VS ACTUAL OVERLAY (LAST 7 DAYS)
                </h3>
                <div style={{ width: '100%', height: 320 }}>
                    {loadingExtra ? (
                        <div className="h-full flex items-center justify-center text-xs text-[#00FFF6] animate-pulse">
                            Generating historical overlay...
                        </div>
                    ) : forecastVsActual ? (
                        <ResponsiveContainer>
                            <LineChart data={overlayChartData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                                <XAxis dataKey="timestamp" stroke="#64748b" tick={{ fontSize: 9, fill: '#64748b' }} />
                                <YAxis stroke="#64748b" tick={{ fontSize: 10, fill: '#64748b' }} tickFormatter={v => `${(v/1000).toFixed(1)}k`} domain={['dataMin - 100', 'dataMax + 100']} />
                                <Tooltip content={<CustomOverlayTooltip />} />
                                <Legend wrapperStyle={{ fontSize: 11 }} />
                                <Line type="monotone" dataKey="Actual" stroke="#00FFF6" strokeWidth={2} dot={false} activeDot={{ r: 4 }} name="Actual" />
                                <Line type="monotone" dataKey="Predicted" stroke="#FF2A6D" strokeWidth={2} dot={false} activeDot={{ r: 4 }} name="Predicted" />
                            </LineChart>
                        </ResponsiveContainer>
                    ) : (
                        <div className="h-full flex flex-col items-center justify-center text-[#8A8F98] space-y-2 text-xs text-center p-4">
                            <TrendingUp className="w-8 h-8 text-slate-600 mb-2" />
                            <p>HISTORICAL DATA NOT AVAILABLE</p>
                            <p className="text-[10px] text-slate-500">Ensure the model is loaded and contains historical baseline load records.</p>
                        </div>
                    )}
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                <div className="bg-[#0B0F1A] border border-[#1e293b] p-4 rounded-lg">
                    <h3 className="text-sm font-bold text-slate-400 mb-2 flex items-center gap-2">
                        <Clock className="w-4 h-4 text-[#00FFF6]" /> HOURLY LOAD PROFILE (24-HOUR PROFILE)
                    </h3>
                    <div style={{ width: '100%', height: 280 }}>
                        <ResponsiveContainer>
                            <AreaChart data={hourlyChartData} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
                                <defs>
                                    <linearGradient id="gG" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="5%" stopColor="#FF2A6D" stopOpacity={0.4}/>
                                        <stop offset="95%" stopColor="#FF2A6D" stopOpacity={0}/>
                                    </linearGradient>
                                </defs>
                                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                                <XAxis dataKey="hour" stroke="#64748b" tick={{ fontSize: 10, fill: '#64748b' }} />
                                <YAxis stroke="#64748b" tick={{ fontSize: 10, fill: '#64748b' }} tickFormatter={v => `${(v/1000).toFixed(1)}k`} domain={['dataMin - 200', 'dataMax + 200']} />
                                <Tooltip contentStyle={{ backgroundColor: '#0B0F1A', border: '1px solid #FF2A6D', borderRadius: 4, fontSize: 11 }} />
                                <Legend wrapperStyle={{ fontSize: 11 }} />
                                <Area type="monotone" dataKey="GRU" stroke="#FF2A6D" fill="url(#gG)" strokeWidth={2} name="Average Hourly Forecast (MW)"
                                    label={(props) => {
                                        const peakHour = hourlyChartData.reduce((iMax, d, i) =>
                                            d.GRU > hourlyChartData[iMax].GRU ? i : iMax, 0);
                                        if (props.index !== peakHour) return null;
                                        return (
                                            <text x={props.x} y={props.y - 8} fill="#FF2A6D" fontSize={10}
                                                fontFamily="monospace" textAnchor="middle">
                                                PEAK: {props.value?.toLocaleString()} MW
                                            </text>
                                        );
                                    }}
                                />
                            </AreaChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                <div className="bg-[#0B0F1A] border border-[#1e293b] p-4 rounded-lg">
                    <h3 className="text-sm font-bold text-slate-400 mb-2 flex items-center gap-2">
                        <TrendingUp className="w-4 h-4 text-[#00FFF6]" /> LOAD DURATION CURVE (LDC)
                    </h3>
                    <div style={{ width: '100%', height: 280 }}>
                        <ResponsiveContainer>
                            <AreaChart data={downsampledLdcData} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
                                <defs>
                                    <linearGradient id="gLdc" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="5%" stopColor="#00FFF6" stopOpacity={0.4}/>
                                        <stop offset="95%" stopColor="#00FFF6" stopOpacity={0}/>
                                    </linearGradient>
                                </defs>
                                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                                <XAxis dataKey="percentage" stroke="#64748b" tick={{ fontSize: 10, fill: '#64748b' }} tickFormatter={v => `${v}%`} />
                                <YAxis stroke="#64748b" tick={{ fontSize: 10, fill: '#64748b' }} tickFormatter={v => `${(v/1000).toFixed(1)}k`} domain={['dataMin - 200', 'dataMax + 200']} />
                                <Tooltip contentStyle={{ backgroundColor: '#0B0F1A', border: '1px solid #00FFF6', borderRadius: 4, fontSize: 11 }} formatter={v => [`${v.toLocaleString()} MW`, 'Load']} labelFormatter={l => `% Time Exceeded: ${l}%`} />
                                <Area type="monotone" dataKey="load" stroke="#00FFF6" fill="url(#gLdc)" strokeWidth={2} name="Load Exceeded (MW)" />
                            </AreaChart>
                        </ResponsiveContainer>
                    </div>
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                <div className="bg-[#0B0F1A] border border-[#1e293b] p-4 rounded-lg">
                    <h3 className="text-sm font-bold text-slate-400 mb-2 flex items-center gap-2">
                        <Activity className="w-4 h-4 text-[#F9F871]" /> LOAD FREQUENCY DISTRIBUTION
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
                        <Target className="w-4 h-4 text-purple-400" /> HISTORICAL FORECAST ERROR TREND
                    </h3>
                    <div style={{ width: '100%', height: 240 }}>
                        {accuracyChartData.length > 0 ? (
                            <ResponsiveContainer>
                                <LineChart data={accuracyChartData} margin={{ top: 20, right: 10, left: -10, bottom: 0 }}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                                    <XAxis dataKey="date" stroke="#64748b" tick={{ fontSize: 9, fill: '#64748b' }} />
                                    <YAxis yAxisId="left" stroke="#a855f7" tick={{ fontSize: 10, fill: '#a855f7' }} tickFormatter={v => `${v}%`} label={{ value: 'MAPE', angle: -90, position: 'insideLeft', fill: '#a855f7', fontSize: 10 }} />
                                    <YAxis yAxisId="right" orientation="right" stroke="#FF2A6D" tick={{ fontSize: 10, fill: '#FF2A6D' }} tickFormatter={v => `${v}`} label={{ value: 'RMSE (MW)', angle: 90, position: 'insideRight', fill: '#FF2A6D', fontSize: 10 }} />
                                    <Tooltip contentStyle={{ backgroundColor: '#0B0F1A', border: '1px solid #a855f7', borderRadius: 4, fontSize: 11 }} />
                                    <Line yAxisId="left" type="monotone" dataKey="MAPE" stroke="#a855f7" strokeWidth={2} activeDot={{ r: 4 }} name="MAPE (%)" />
                                    <Line yAxisId="right" type="monotone" dataKey="RMSE" stroke="#FF2A6D" strokeWidth={2} activeDot={{ r: 4 }} name="RMSE (MW)" />
                                </LineChart>
                            </ResponsiveContainer>
                        ) : (
                            <div className="h-full flex flex-col items-center justify-center text-[#8A8F98] space-y-2 text-xs text-center p-4">
                                <TrendingUp className="w-8 h-8 text-slate-600 mb-2" />
                                <p>NO HISTORICAL ACCURACY DATA</p>
                                <p className="text-[10px] text-slate-500">Run model retraining or feed actual historical data to enable error trends.</p>
                            </div>
                        )}
                    </div>
                </div>

                <div className="bg-[#0B0F1A] border border-[#1e293b] p-4 rounded-lg">
                    <h3 className="text-sm font-bold text-slate-400 mb-2 flex items-center gap-2">
                        <Target className="w-4 h-4 text-green-400" /> DIURNAL DEMAND PROFILE SHARE
                    </h3>
                    <div style={{ width: '100%', height: 240 }}>
                        <ResponsiveContainer>
                            <PieChart>
                                <Pie data={pieData} cx="50%" cy="50%" innerRadius={45} outerRadius={75} paddingAngle={5} dataKey="value">
                                    {pieData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                                </Pie>
                                <Tooltip contentStyle={{ backgroundColor: '#0B0F1A', border: '1px solid #00FFF6', borderRadius: 4, fontSize: 11 }} formatter={(v) => [`${v.toLocaleString()} MW`, 'Avg Load']} />
                                <Legend wrapperStyle={{ fontSize: 10 }} />
                            </PieChart>
                        </ResponsiveContainer>
                    </div>
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                {/* Feature Importance Chart */}
                <div className="lg:col-span-2 bg-[#0B0F1A] border border-[#1e293b] p-4 rounded-lg">
                    <h3 className="text-sm font-bold text-slate-400 mb-3 flex items-center gap-2">
                        <Cpu className="w-4 h-4 text-purple-400" /> LIGHTGBM FEATURE IMPORTANCE (GAIN)
                    </h3>
                    <div style={{ width: '100%', height: 320 }} className="font-mono">
                        {loadingExtra ? (
                            <div className="h-full flex items-center justify-center text-xs text-[#00FFF6] animate-pulse">
                                Loading feature importances...
                            </div>
                        ) : featureImportance.length > 0 ? (
                            <ResponsiveContainer>
                                <BarChart
                                    data={featureImportance}
                                    layout="vertical"
                                    margin={{ top: 5, right: 10, left: 15, bottom: 5 }}
                                >
                                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                                    <XAxis type="number" stroke="#64748b" tick={{ fontSize: 9, fill: '#64748b' }} tickFormatter={v => `${v}%`} />
                                    <YAxis dataKey="feature" type="category" stroke="#64748b" tick={{ fontSize: 8, fill: '#64748b' }} width={120} />
                                    <Tooltip contentStyle={{ backgroundColor: '#0B0F1A', border: '1px solid #a855f7', borderRadius: 4, fontSize: 10 }} formatter={v => [`${v.toFixed(2)}%`, 'Relative Gain']} />
                                    <Bar dataKey="importance" fill="#a855f7" radius={[0, 4, 4, 0]}>
                                        {featureImportance.map((entry, index) => (
                                            <Cell key={`cell-${index}`} fill={index % 2 === 0 ? '#a855f7' : '#FF2A6D'} />
                                        ))}
                                    </Bar>
                                </BarChart>
                            </ResponsiveContainer>
                        ) : (
                            <div className="h-full flex items-center justify-center text-xs text-slate-500">
                                Feature importances not available
                            </div>
                        )}
                    </div>
                </div>

                {/* Training & Performance Stats */}
                <div className="space-y-4">
                    {metrics && (
                        <>
                            <div className="bg-gradient-to-r from-[#0B0F1A] to-[#131b2d] border border-[#00FFF6]/20 p-4 rounded-lg">
                                <h3 className="text-sm font-bold text-[#00FFF6] mb-3 flex items-center gap-2">
                                    <Database className="w-4 h-4" /> TRAINING DATA
                                </h3>
                                <div className="space-y-2 text-xs">
                                    <div className="flex justify-between"><span className="text-slate-500">Samples</span><span className="text-white">{(metrics.gru_hybrid.training_samples || 0).toLocaleString()}</span></div>
                                    <div className="flex justify-between"><span className="text-slate-500">Features</span><span className="text-white">{metrics.gru_hybrid.features}</span></div>
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
                        </>
                    )}
                </div>
            </div>

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
