import { useState, useEffect } from 'react';
import { useData } from '../contexts/DataContext';
import clsx from 'clsx';
import { predictLoad, getRealtimeStatus } from '../services/api';
import ForecastChart from '../components/ForecastChart';
import ControlPanel from '../components/ControlPanel';
import { AlertCircle, Maximize2, Minimize2, Radio, Zap, BarChart3, Sun, Wind, Activity, Flame, Droplets, TrendingUp } from 'lucide-react';

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

    // Fix 22: Pause realtime polling when the browser tab is in the background
    useEffect(() => {
        const fetchStatus = async () => {
            if (document.visibilityState === 'hidden') return;
            try {
                const status = await getRealtimeStatus();
                setRealtimeStatus(status);
            } catch (err) {
                console.error("Failed to fetch realtime status", err);
            }
        };

        fetchStatus();
        const interval = setInterval(fetchStatus, 300000);
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

    // Fix 23: Filter null/NaN before Math.max/min to prevent -Infinity/NaN on stats cards
    const getModelStats = () => {
        if (!data || !data.loads_lightgbm_gru || data.loads_lightgbm_gru.length === 0) return null;
        const gruLoads = data.loads_lightgbm_gru.filter(v => v !== null && !isNaN(v));
        if (!gruLoads.length) return null;

        const gruMax = Math.max(...gruLoads);
        const gruMin = Math.min(...gruLoads);
        const gruAvg = gruLoads.reduce((a, b) => a + b, 0) / gruLoads.length;

        return { gruMax, gruMin, gruAvg };
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
                <div className="bg-[#0B0F1A] border border-[#00FFF6]/20 p-4 md:p-5 animate-in fade-in duration-500 space-y-4">
                    <div className="flex items-center justify-between border-b border-[#00FFF6]/10 pb-3 flex-wrap gap-2">
                        <div className="flex items-center gap-2">
                            <Radio className="w-4 h-4 text-[#00FFF6] animate-pulse" />
                            <span className="text-xs font-mono text-[#00FFF6] uppercase tracking-widest font-bold">
                                {realtimeStatus.source === 'kptcl_sldc_live' ? 'LIVE TELEMETRY // KARNATAKA GRID' : 'LIVE_GRID_STATUS'}
                            </span>
                            <span className="px-2 py-0.5 border border-green-500/30 bg-green-500/10 text-[9px] font-mono text-green-400 uppercase rounded-sm">
                                {realtimeStatus.source === 'kptcl_sldc_live' ? 'REAL-TIME' : 'STALE / FALLBACK'}
                            </span>
                        </div>
                        <span className="text-xs font-mono text-[#8A8F98]">AS_OF: {realtimeStatus.timestamp}</span>
                    </div>

                    {/* Primary Grid Metrics */}
                    <div className="grid grid-cols-2 sm:grid-cols-2 md:grid-cols-4 xl:grid-cols-4 gap-3">
                        <div className="bg-[#0f1626] p-3 border border-[#1e293b] flex flex-col justify-between">
                            <span className="text-[10px] font-mono text-slate-500 uppercase tracking-wider">Bengaluru Load (BESCOM)</span>
                            <span className="text-xl md:text-2xl font-mono font-bold text-[#00FFF6] mt-2">
                                {formatNullable(realtimeStatus.bescom_mw || realtimeStatus.current_load_mw, ' MW')}
                            </span>
                        </div>
                        <div className="bg-[#0f1626] p-3 border border-[#1e293b] flex flex-col justify-between">
                            <span className="text-[10px] font-mono text-slate-500 uppercase tracking-wider">Karnataka Demand</span>
                            <span className="text-xl md:text-2xl font-mono font-bold text-white mt-2">
                                {formatNullable(realtimeStatus.state_demand_mw || realtimeStatus.current_load_mw, ' MW')}
                            </span>
                        </div>
                        <div className="bg-[#0f1626] p-3 border border-[#1e293b] flex flex-col justify-between">
                            <span className="text-[10px] font-mono text-slate-500 uppercase tracking-wider">Grid Frequency</span>
                            <span className="text-xl md:text-2xl font-mono font-bold text-[#F9F871] mt-2">
                                {formatNullable(realtimeStatus.frequency_hz, ' Hz')}
                            </span>
                        </div>
                        <div className="bg-[#0f1626] p-3 border border-[#1e293b] flex flex-col justify-between">
                            <span className="text-[10px] font-mono text-slate-500 uppercase tracking-wider">State UI / Deviation</span>
                            <span className={`text-xl md:text-2xl font-mono font-bold mt-2 ${
                                realtimeStatus.od_ud_mw === null || realtimeStatus.od_ud_mw === undefined 
                                    ? 'text-slate-500' 
                                    : realtimeStatus.od_ud_mw >= 0 
                                    ? 'text-green-400' 
                                    : 'text-red-400'
                            }`}>
                                {realtimeStatus.od_ud_mw !== null ? `${realtimeStatus.od_ud_mw >= 0 ? '+' : ''}${realtimeStatus.od_ud_mw} MW` : 'N/A'}
                            </span>
                        </div>
                    </div>

                    {/* Secondary Metrics / Breakdown Panels */}
                    {realtimeStatus.source === 'kptcl_sldc_live' && (
                        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 border-t border-[#00FFF6]/10 pt-4">
                            {/* Generation Breakdown */}
                            {realtimeStatus.generation_breakdown && (
                                <div className="space-y-3">
                                    <div className="flex items-center gap-1.5 text-xs font-mono text-[#8A8F98] uppercase tracking-wider">
                                        <Activity className="w-3.5 h-3.5 text-[#F9F871]" />
                                        <span>Live Generation Breakdown ({realtimeStatus.generation_mw ?? '—'} MW)</span>
                                    </div>
                                    <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
                                        {(() => {
                                            const gb = realtimeStatus.generation_breakdown;
                                            const total = (gb.solar_mw + gb.wind_mw + gb.thermal_mw + gb.thermal_ipp_mw + gb.hydro_mw + gb.other_mw) || 1;
                                            const items = [
                                                { label: 'Solar', val: gb.solar_mw, icon: <Sun className="w-3 h-3 text-[#F9F871]" />, color: 'bg-[#F9F871]/20 border-[#F9F871]/40 text-[#F9F871]' },
                                                { label: 'Wind', val: gb.wind_mw, icon: <Wind className="w-3 h-3 text-emerald-400" />, color: 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400' },
                                                { label: 'Thermal', val: gb.thermal_mw + gb.thermal_ipp_mw, icon: <Flame className="w-3 h-3 text-orange-400" />, color: 'bg-orange-500/10 border-orange-500/30 text-orange-400' },
                                                { label: 'Hydro', val: gb.hydro_mw, icon: <Droplets className="w-3 h-3 text-blue-400" />, color: 'bg-blue-500/10 border-blue-500/30 text-blue-400' },
                                                { label: 'Other', val: gb.other_mw, icon: <Zap className="w-3 h-3 text-purple-400" />, color: 'bg-purple-500/10 border-purple-500/30 text-purple-400' },
                                            ];
                                            return items.map(item => (
                                                <div key={item.label} className={`p-2 border rounded-sm flex items-center justify-between ${item.color.split(' ')[0]} ${item.color.split(' ')[1]}`}>
                                                    <div className="flex items-center gap-1">
                                                        {item.icon}
                                                        <span className="text-[10px] font-mono font-bold uppercase">{item.label}</span>
                                                    </div>
                                                    <div className="text-right">
                                                        <p className="text-[11px] font-mono font-bold">{item.val} MW</p>
                                                        <p className="text-[8px] font-mono opacity-60">{Math.round((item.val / total) * 100)}%</p>
                                                    </div>
                                                </div>
                                            ));
                                        })()}
                                    </div>
                                </div>
                            )}

                            {/* ESCOM wise Drawal Breakdown */}
                            <div className="space-y-3">
                                <div className="flex items-center gap-1.5 text-xs font-mono text-[#8A8F98] uppercase tracking-wider">
                                    <TrendingUp className="w-3.5 h-3.5 text-[#00FFF6]" />
                                    <span>ESCOM Drawal Distribution</span>
                                </div>
                                <div className="grid grid-cols-2 sm:grid-cols-5 gap-2">
                                    {[
                                        { label: 'BESCOM (Blr)', val: realtimeStatus.bescom_mw },
                                        { label: 'HESCOM (Hbl)', val: realtimeStatus.hescom_mw },
                                        { label: 'GESCOM (Glb)', val: realtimeStatus.gescom_mw },
                                        { label: 'CESC (Mys)', val: realtimeStatus.cesc_mw },
                                        { label: 'MESCOM (Mng)', val: realtimeStatus.mescom_mw },
                                    ].map(escom => (
                                        <div key={escom.label} className="p-2 border border-[#1e293b] bg-[#0f1626] rounded-sm text-center">
                                            <p className="text-[9px] font-mono text-slate-500 uppercase font-bold whitespace-nowrap">{escom.label}</p>
                                            <p className="text-xs font-mono font-bold text-white mt-1">{escom.val !== null ? `${escom.val} MW` : 'N/A'}</p>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </div>
                    )}
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
                            {/* GRU Model Stats */}
                            <div className="bg-[#0B0F1A] border border-[#FF2A6D]/30 p-4">
                                <h4 className="text-xs font-mono text-[#FF2A6D] uppercase mb-3 flex items-center gap-2">
                                    <BarChart3 className="w-4 h-4" /> GRU_HYBRID FORECAST
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
