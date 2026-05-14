import { useState, useEffect } from 'react';
import { useData } from '../contexts/DataContext';
import clsx from 'clsx';
import { predictLoad, getRealtimeStatus } from '../services/api';
import ForecastChart from '../components/ForecastChart';
import ControlPanel from '../components/ControlPanel';
import StatsCard from '../components/StatsCard';
import { Activity, Zap, TrendingUp, AlertCircle, Maximize2, Minimize2, Radio, TrendingDown, BarChart3 } from 'lucide-react';

const formatNullable = (value, suffix = '') => {
    if (value === null || value === undefined) {
        return 'N/A';
    }
    return `${value}${suffix}`;
};

const Dashboard = () => {
    const { data, setData, realtimeStatus, setRealtimeStatus } = useData();
    const [loading, setLoading] = useState(false);
    const [isFullScreen, setIsFullScreen] = useState(false);
    const [error, setError] = useState(null);

    // Fetch real-time status every 10 seconds
    useEffect(() => {
        const fetchStatus = async () => {
            try {
                const status = await getRealtimeStatus();
                setRealtimeStatus(status);
            } catch (error) {
                console.error("Failed to fetch realtime status", error);
            }
        };
        
        fetchStatus();
        const interval = setInterval(fetchStatus, 10000);
        return () => clearInterval(interval);
    }, [setRealtimeStatus]);

    const handlePredict = async (startDate, endDate) => {
        setLoading(true);
        setError(null);
        try {
            const result = await predictLoad(startDate, endDate);
            setData(result);
        } catch (err) {
            console.error("Prediction Error:", err);
            let errorMessage = "Failed to fetch forecast. Ensure backend is running.";

            if (err.response?.data?.detail) {
                const detail = err.response.data.detail;
                if (typeof detail === 'string') {
                    errorMessage = detail;
                } else if (Array.isArray(detail)) {
                    errorMessage = detail.map(e => e.msg).join(', ');
                } else {
                    errorMessage = JSON.stringify(detail);
                }
            }
            setError(errorMessage);
        } finally {
            setLoading(false);
        }
    };

    // Calculate model comparison stats
    const getModelStats = () => {
        if (!data || !data.loads_lightgbm_lstm || data.loads_lightgbm_lstm.length === 0) return null;
        const lstmLoads = data.loads_lightgbm_lstm;
        const gruLoads = data.loads_lightgbm_gru;
        
        const lstmMax = Math.max(...lstmLoads);
        const lstmMin = Math.min(...lstmLoads);
        const lstmAvg = lstmLoads.reduce((a, b) => a + b, 0) / lstmLoads.length;
        
        const gruMax = Math.max(...gruLoads);
        const gruMin = Math.min(...gruLoads);
        const gruAvg = gruLoads.reduce((a, b) => a + b, 0) / gruLoads.length;
        
        return { lstmMax, lstmMin, lstmAvg, gruMax, gruMin, gruAvg };
    };

    const modelStats = getModelStats();

    return (
        <div className="space-y-6">
            <header className="mb-6">
                <h2 className="text-2xl md:text-3xl font-bold text-white mb-2">Dashboard</h2>
                <p className="text-slate-400 text-sm">Real-time energy load forecasting and grid analytics for Bengaluru BESCOM.</p>
            </header>

            {/* Real-time Grid Status Bar */}
            {realtimeStatus && (
                <div className="bg-[#0B0F1A] border border-[#00FFF6]/30 p-3 md:p-4 animate-in fade-in duration-500">
                    <div className="flex items-center gap-2 mb-3">
                        <Radio className="w-4 h-4 text-green-400 animate-pulse" />
                        <span className="text-xs font-mono text-[#00FFF6] uppercase tracking-wider">LIVE_GRID_STATUS</span>
                        <span className="text-xs text-slate-500 ml-auto">{realtimeStatus.timestamp}</span>
                    </div>
                    <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-7 gap-2 md:gap-3 text-center">
                        <div className="bg-[#131b2d] p-2 md:p-3 border border-[#1e293b]">
                            <p className="text-[9px] md:text-[10px] text-slate-500 uppercase">Frequency</p>
                            <p className="text-sm md:text-lg font-mono text-[#00FFF6]">{formatNullable(realtimeStatus.frequency_hz, ' Hz')}</p>
                        </div>
                        <div className="bg-[#131b2d] p-2 md:p-3 border border-[#1e293b]">
                            <p className="text-[9px] md:text-[10px] text-slate-500 uppercase">Current Load</p>
                            <p className="text-sm md:text-lg font-mono text-white">{formatNullable(realtimeStatus.current_load_mw, ' MW')}</p>
                        </div>
                        <div className="bg-[#131b2d] p-2 md:p-3 border border-[#1e293b]">
                            <p className="text-[9px] md:text-[10px] text-slate-500 uppercase">Schedule</p>
                            <p className="text-sm md:text-lg font-mono text-slate-300">{formatNullable(realtimeStatus.schedule_mw, ' MW')}</p>
                        </div>
                        <div className="bg-[#131b2d] p-2 md:p-3 border border-[#1e293b]">
                            <p className="text-[9px] md:text-[10px] text-slate-500 uppercase">Drawal</p>
                            <p className="text-sm md:text-lg font-mono text-slate-300">{formatNullable(realtimeStatus.drawal_mw, ' MW')}</p>
                        </div>
                        <div className="bg-[#131b2d] p-2 md:p-3 border border-[#1e293b]">
                            <p className="text-[9px] md:text-[10px] text-slate-500 uppercase">OD/UD</p>
                            <p className={`text-sm md:text-lg font-mono ${realtimeStatus.od_ud_mw === null || realtimeStatus.od_ud_mw === undefined ? 'text-slate-500' : realtimeStatus.od_ud_mw >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                                {formatNullable(realtimeStatus.od_ud_mw, ' MW')}
                            </p>
                        </div>
                        <div className="bg-[#131b2d] p-2 md:p-3 border border-[#1e293b]">
                            <p className="text-[9px] md:text-[10px] text-slate-500 uppercase">Today Max</p>
                            <p className="text-sm md:text-lg font-mono text-[#FF2A6D]">{formatNullable(realtimeStatus.today_max?.value, ' MW')}</p>
                        </div>
                        <div className="bg-[#131b2d] p-2 md:p-3 border border-[#1e293b]">
                            <p className="text-[9px] md:text-[10px] text-slate-500 uppercase">Today Min</p>
                            <p className="text-sm md:text-lg font-mono text-[#00FFF6]">{formatNullable(realtimeStatus.today_min?.value, ' MW')}</p>
                        </div>
                    </div>
                </div>
            )}

            <div className="grid grid-cols-1 lg:grid-cols-4 gap-4 md:gap-6">
                <div className="lg:col-span-1 space-y-4 md:space-y-6 animate-in slide-in-from-left duration-700 fade-in delay-100">
                    <ControlPanel onPredict={handlePredict} loading={loading} />

                    {error && (
                        <div className="bg-red-500/10 border border-red-500/20 p-4 rounded-xl flex items-start space-x-3">
                            <AlertCircle className="w-5 h-5 text-red-500 shrink-0 mt-0.5" />
                            <p className="text-sm text-red-200">{error}</p>
                        </div>
                    )}

                    {data?.used_dummy && (
                        <div className="bg-amber-500/10 border border-amber-500/30 p-4 rounded-xl flex items-start space-x-3">
                            <AlertCircle className="w-5 h-5 text-amber-400 shrink-0 mt-0.5" />
                            <div className="text-sm text-amber-200">
                                <p className="font-semibold">Dummy mode active</p>
                                <p>
                                    Forecast is using fallback prediction instead of the trained model.
                                    {data?.dummy_reason ? ` Reason: ${data.dummy_reason}` : ''}
                                </p>
                            </div>
                        </div>
                    )}

                    {modelStats && (
                        <div className="space-y-3">
                            {/* LSTM Model Stats */}
                            <div className="bg-[#0B0F1A] border border-[#00FFF6]/30 p-4">
                                <h4 className="text-xs font-mono text-[#00FFF6] uppercase mb-3 flex items-center gap-2">
                                    <BarChart3 className="w-4 h-4" /> LSTM_HYBRID
                                </h4>
                                <div className="grid grid-cols-3 gap-2 text-center">
                                    <div>
                                        <p className="text-[10px] text-slate-500">PEAK</p>
                                        <p className="text-sm font-mono text-[#00FFF6]">{modelStats.lstmMax.toFixed(0)}</p>
                                    </div>
                                    <div>
                                        <p className="text-[10px] text-slate-500">MIN</p>
                                        <p className="text-sm font-mono text-slate-300">{modelStats.lstmMin.toFixed(0)}</p>
                                    </div>
                                    <div>
                                        <p className="text-[10px] text-slate-500">AVG</p>
                                        <p className="text-sm font-mono text-slate-300">{modelStats.lstmAvg.toFixed(0)}</p>
                                    </div>
                                </div>
                            </div>
                            
                            {/* GRU Model Stats */}
                            <div className="bg-[#0B0F1A] border border-[#FF2A6D]/30 p-4">
                                <h4 className="text-xs font-mono text-[#FF2A6D] uppercase mb-3 flex items-center gap-2">
                                    <BarChart3 className="w-4 h-4" /> GRU_HYBRID
                                </h4>
                                <div className="grid grid-cols-3 gap-2 text-center">
                                    <div>
                                        <p className="text-[10px] text-slate-500">PEAK</p>
                                        <p className="text-sm font-mono text-[#FF2A6D]">{modelStats.gruMax.toFixed(0)}</p>
                                    </div>
                                    <div>
                                        <p className="text-[10px] text-slate-500">MIN</p>
                                        <p className="text-sm font-mono text-slate-300">{modelStats.gruMin.toFixed(0)}</p>
                                    </div>
                                    <div>
                                        <p className="text-[10px] text-slate-500">AVG</p>
                                        <p className="text-sm font-mono text-slate-300">{modelStats.gruAvg.toFixed(0)}</p>
                                    </div>
                                </div>
                            </div>

                            {/* Model Difference */}
                            <div className="bg-[#0B0F1A] border border-[#F9F871]/30 p-4">
                                <h4 className="text-xs font-mono text-[#F9F871] uppercase mb-3 flex items-center gap-2">
                                    <TrendingUp className="w-4 h-4" /> MODEL_DIFF
                                </h4>
                                <div className="grid grid-cols-2 gap-2 text-center">
                                    <div>
                                        <p className="text-[10px] text-slate-500">PEAK DIFF</p>
                                        <p className="text-sm font-mono text-[#F9F871]">
                                            {Math.abs(modelStats.lstmMax - modelStats.gruMax).toFixed(0)} MW
                                        </p>
                                    </div>
                                    <div>
                                        <p className="text-[10px] text-slate-500">AVG DIFF</p>
                                        <p className="text-sm font-mono text-[#F9F871]">
                                            {Math.abs(modelStats.lstmAvg - modelStats.gruAvg).toFixed(0)} MW
                                        </p>
                                    </div>
                                </div>
                            </div>
                        </div>
                    )}
                </div>

                <div className={clsx(
                    "animate-in slide-in-from-right fade-in delay-200 transition-all duration-500",
                    isFullScreen ? "fixed inset-0 z-50 p-4 bg-[#0B0F1A]" : "lg:col-span-3"
                )}>
                    <div className={clsx(
                        "bg-[#0B0F1A] border border-[#1e293b] p-4 md:p-6 relative overflow-hidden group transition-all duration-500",
                        isFullScreen ? "h-full w-full" : "h-[500px] md:h-[600px]"
                    )}>
                        {/* Grid & Scanner Background Effect */}
                        <div className="absolute inset-0 bg-[linear-gradient(rgba(0,255,246,0.03)_1px,transparent_1px),linear-gradient(90deg,rgba(0,255,246,0.03)_1px,transparent_1px)] bg-[size:40px_40px]" />
                        <div className="absolute inset-0 bg-scanner pointer-events-none opacity-20" />

                        {/* Full Screen Toggle */}
                        <button
                            onClick={() => setIsFullScreen(!isFullScreen)}
                            className="absolute top-4 right-4 z-20 p-2 bg-[#0B0F1A]/80 border border-[#00FFF6]/30 text-[#00FFF6] hover:bg-[#00FFF6]/10 hover:border-[#00FFF6] transition-all rounded-sm group-hover:opacity-100 opacity-0"
                            title={isFullScreen ? "Exit Fullscreen" : "Enter Fullscreen"}
                        >
                            {isFullScreen ? <Minimize2 className="w-5 h-5" /> : <Maximize2 className="w-5 h-5" />}
                        </button>

                        {data && data.timestamps && data.timestamps.length > 0 ? (
                            <ForecastChart data={data} />
                        ) : data ? (
                            <div className="h-full flex flex-col items-center justify-center text-[#8A8F98] space-y-4 font-mono uppercase tracking-widest text-xs">
                                <div className="p-4 bg-[#131b2d] border border-[#1e293b] shadow-[0_0_15px_rgba(0,0,0,0.5)]">
                                    <AlertCircle className="w-8 h-8 text-[#FF2A6D]" />
                                </div>
                                <p>NO_DATA_AVAILABLE</p>
                                <p className="text-[10px] text-slate-600">The selected date range returned empty results.</p>
                            </div>
                        ) : (
                            <div className="h-full flex flex-col items-center justify-center text-[#8A8F98] space-y-4 font-mono uppercase tracking-widest text-xs animate-pulse">
                                <div className="p-4 bg-[#131b2d] border border-[#1e293b] shadow-[0_0_15px_rgba(0,0,0,0.5)]">
                                    <Zap className="w-8 h-8 text-[#8A8F98]" />
                                </div>
                                <p>AWAITING_INPUT_PARAMETERS...</p>
                                <p className="text-[10px] text-slate-600">Select date range and click Generate Forecast</p>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Dashboard;
