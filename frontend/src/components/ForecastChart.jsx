import React from 'react';
import {
    XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
    Brush, Legend, ReferenceLine, ReferenceDot, ComposedChart, Area, Line
} from 'recharts';

// CustomTooltip defined OUTSIDE the component so it is not re-created on every render (prevents flickering).
// CustomTooltip defined OUTSIDE the component so it is not re-created on every render (prevents flickering).
const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
        const gruItem = payload.find(p => p.dataKey === 'gru_hybrid');
        const gruVal = gruItem ? gruItem.value : null;

        return (
            <div className="bg-[#0B0F1A] border border-[#FF2A6D]/50 p-3 font-mono text-xs shadow-lg">
                <p className="text-slate-400 mb-2 border-b border-[#1e293b] pb-1">{label}</p>
                <div className="space-y-1">
                    {gruVal !== null && (
                        <p className="flex justify-between gap-4">
                            <span className="text-[#FF2A6D]">Forecast:</span>
                            <span className="text-white font-bold">{gruVal.toLocaleString()} MW</span>
                        </p>
                    )}
                </div>
            </div>
        );
    }
    return null;
};

const ForecastChart = ({ data }) => {
    if (!data || !data.timestamps) return null;

    const hasGru = data.loads_lightgbm_gru && data.loads_lightgbm_gru.some(v => v !== null && v !== undefined);

    // Process data with better time formatting
    const chartData = data.timestamps.map((ts, i) => {
        const gruVal = data.loads_lightgbm_gru ? data.loads_lightgbm_gru[i] : null;
        
        // Extract time for display
        const timePart = ts.split(' ')[1] || ts;
        const datePart = ts.split(' ')[0] || '';

        return {
            time: ts,
            displayTime: timePart.substring(0, 5), // HH:MM
            date: datePart,
            gru_hybrid: gruVal !== null && gruVal !== undefined ? Math.round(gruVal) : null,
            avg: gruVal !== null && gruVal !== undefined ? Math.round(gruVal) : 0
        };
    });

    // Calculate stats safely
    const avgLoad = chartData.length ? chartData.reduce((sum, d) => sum + d.avg, 0) / chartData.length : 0;
    
    const peakLoad = chartData.length 
        ? Math.max(...chartData.map(d => d.gru_hybrid).filter(v => v !== null))
        : 0;

    const minLoad = chartData.length
        ? Math.min(...chartData.map(d => d.gru_hybrid).filter(v => v !== null))
        : 0;

    const targetKey = 'gru_hybrid';
    const peakIdx = chartData.length ? chartData.reduce((iMax, d, i) => (d[targetKey] || 0) > (chartData[iMax][targetKey] || 0) ? i : iMax, 0) : 0;
    const minIdx  = chartData.length ? chartData.reduce((iMin, d, i) => (d[targetKey] || Infinity) < (chartData[iMin][targetKey] || Infinity) ? i : iMin, 0) : 0;
    const peakPoint = chartData.length ? chartData[peakIdx] : null;
    const minPoint  = chartData.length ? chartData[minIdx] : null;

    // Sample data for X-axis (show every 2 hours for readability)
    const tickFormatter = (value, index) => {
        if (chartData.length <= 48) return value.split(' ')[1]?.substring(0, 5) || value;
        // For longer ranges, show less ticks
        if (index % Math.ceil(chartData.length / 12) === 0) {
            return value.split(' ')[1]?.substring(0, 5) || value;
        }
        return '';
    };

    return (
        <div className="w-full h-full flex flex-col">
            <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-slate-300">Hybrid Stacking Forecast Profile</h3>
                <div className="flex gap-4 text-xs font-mono">
                    <span className="text-slate-500">Peak: <span className="text-[#FF2A6D]">{peakLoad.toLocaleString()} MW</span></span>
                    <span className="text-slate-500">Min: <span className="text-[#00FFF6]">{minLoad.toLocaleString()} MW</span></span>
                    <span className="text-slate-500">Avg: <span className="text-slate-300">{Math.round(avgLoad).toLocaleString()} MW</span></span>
                </div>
            </div>
            <ResponsiveContainer width="100%" height="90%">
                <ComposedChart data={chartData} margin={{ top: 20, right: 10, left: 0, bottom: 0 }}>
                    <defs>
                        <linearGradient id="colorGRU" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#FF2A6D" stopOpacity={0.3} />
                            <stop offset="95%" stopColor="#FF2A6D" stopOpacity={0} />
                        </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                    <XAxis
                        dataKey="time"
                        stroke="#8A8F98"
                        tick={{ fill: '#8A8F98', fontFamily: 'monospace', fontSize: 9 }}
                        tickFormatter={tickFormatter}
                        interval="preserveStartEnd"
                        minTickGap={30}
                        axisLine={{ stroke: '#1e293b' }}
                    />
                    <YAxis
                        stroke="#8A8F98"
                        tick={{ fill: '#8A8F98', fontFamily: 'monospace', fontSize: 9 }}
                        domain={[Math.floor(minLoad * 0.95 / 100) * 100, Math.ceil(peakLoad * 1.02 / 100) * 100]}
                        tickFormatter={(value) => `${(value/1000).toFixed(1)}k`}
                        axisLine={{ stroke: '#1e293b' }}
                        width={45}
                    />
                    <Tooltip content={<CustomTooltip />} />
                    <Legend
                        wrapperStyle={{ paddingTop: '10px', fontFamily: 'monospace', fontSize: '11px' }}
                        formatter={(value) => <span className="text-slate-300">{value}</span>}
                    />

                    {/* Reference line for average */}
                    <ReferenceLine
                        y={avgLoad}
                        stroke="#F9F871"
                        strokeDasharray="5 5"
                        strokeOpacity={0.5}
                        label={{ value: 'AVG', position: 'right', fill: '#F9F871', fontSize: 9, fontFamily: 'monospace' }}
                    />

                    {/* Peak annotation */}
                    {peakPoint && peakPoint[targetKey] !== null && (
                        <ReferenceDot
                            x={peakPoint.time} y={peakPoint[targetKey]}
                            r={4} fill="#FF2A6D" stroke="#0B0F1A" strokeWidth={2}
                            label={{ value: `▲ ${peakPoint[targetKey].toLocaleString()} MW`, position: 'top', fill: '#FF2A6D', fontSize: 10, fontFamily: 'monospace' }}
                        />
                    )}

                    {/* Minimum annotation */}
                    {minPoint && minPoint[targetKey] !== null && (
                        <ReferenceDot
                            x={minPoint.time} y={minPoint[targetKey]}
                            r={4} fill="#00FFF6" stroke="#0B0F1A" strokeWidth={2}
                            label={{ value: `▼ ${minPoint[targetKey].toLocaleString()} MW`, position: 'bottom', fill: '#00FFF6', fontSize: 10, fontFamily: 'monospace' }}
                        />
                    )}

                    {/* GRU Area (Cyberpunk styled gradient fill) */}
                    {hasGru && (
                        <Area
                            type="monotone"
                            dataKey="gru_hybrid"
                            name="LightGBM + GRU Stacking"
                            stroke="#FF2A6D"
                            fillOpacity={1}
                            fill="url(#colorGRU)"
                            strokeWidth={2}
                            dot={false}
                            activeDot={{ r: 4, fill: '#FF2A6D', stroke: '#0B0F1A', strokeWidth: 2 }}
                        />
                    )}

                    {/* Brush */}
                    <Brush
                        dataKey="time"
                        height={25}
                        stroke="#00FFF6"
                        fill="#0B0F1A"
                        tickFormatter={(val) => {
                            const parts = val.split(' ');
                            return parts.length > 1 ? `${parts[0].slice(5)} ${parts[1].substring(0, 5)}` : val;
                        }}
                        travellerWidth={8}
                    />
                </ComposedChart>
            </ResponsiveContainer>
        </div>
    );
};

export default ForecastChart;
