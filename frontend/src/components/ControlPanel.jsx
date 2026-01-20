import { useState } from 'react';
import { Calendar, Clock, ArrowRight, Activity } from 'lucide-react';

const ControlPanel = ({ onPredict, loading }) => {
    // Use current date dynamically
    const today = new Date();
    const tomorrow = new Date(today);
    tomorrow.setDate(today.getDate() + 1);
    
    const formatDate = (d) => d.toISOString().split('T')[0];
    
    const [startDate, setStartDate] = useState(formatDate(today));
    const [endDate, setEndDate] = useState(formatDate(tomorrow));
    const oneWeekAhead = new Date(today.getFullYear(), today.getMonth(), today.getDate() + 7);

    const handleSubmit = (e) => {
        e.preventDefault();
        // Enforce constraints: start <= today, end between start and today+7
        const s = new Date(startDate);
        const eDate = new Date(endDate);
        const todayOnly = new Date(today.getFullYear(), today.getMonth(), today.getDate());
        const maxEnd = new Date(oneWeekAhead.getFullYear(), oneWeekAhead.getMonth(), oneWeekAhead.getDate());

        // Clamp start to today if future
        if (s > todayOnly) {
            setStartDate(formatDate(todayOnly));
        }
        // Ensure end >= start and end <= today+7
        let nextEnd = eDate < s ? s : eDate;
        if (nextEnd > maxEnd) nextEnd = maxEnd;
        setEndDate(formatDate(nextEnd));

        onPredict(formatDate(s > todayOnly ? todayOnly : s), formatDate(nextEnd));
    };

    return (
        <div className="bg-[#0B0F1A] border border-[#00FFF6]/30 p-6 shadow-[0_0_20px_rgba(0,255,246,0.05)] relative group">
            {/* Corner Accents */}
            <div className="absolute top-0 left-0 w-2 h-2 border-t border-l border-[#00FFF6]" />
            <div className="absolute top-0 right-0 w-2 h-2 border-t border-r border-[#00FFF6]" />
            <div className="absolute bottom-0 left-0 w-2 h-2 border-b border-l border-[#00FFF6]" />
            <div className="absolute bottom-0 right-0 w-2 h-2 border-b border-r border-[#00FFF6]" />

            <h3 className="text-lg font-mono font-semibold mb-6 text-[#00FFF6] tracking-widest uppercase border-b border-[#00FFF6]/20 pb-2">
                // SYSTEM_CONFIG
            </h3>

            <form onSubmit={handleSubmit} className="space-y-6">
                <div className="space-y-2">
                    <label className="text-xs font-mono text-[#00FFF6]/70 flex items-center gap-2 uppercase tracking-wider">
                        <Calendar className="w-3 h-3" /> Start_Date
                    </label>
                    <input
                        type="date"
                        value={startDate}
                        onChange={(e) => setStartDate(e.target.value)}
                        className="w-full bg-[#131b2d] border border-[#00FFF6]/50 px-4 py-3 text-[#EAEAEA] font-mono focus:border-[#00FFF6] focus:shadow-[0_0_10px_rgba(0,255,246,0.2)] outline-none transition-all placeholder-[#8A8F98]"
                        max={formatDate(today)}
                    />
                </div>

                <div className="space-y-2">
                    <label className="text-xs font-mono text-[#FF2A6D]/70 flex items-center gap-2 uppercase tracking-wider">
                        <Clock className="w-3 h-3" /> End_Date
                    </label>
                    <input
                        type="date"
                        value={endDate}
                        onChange={(e) => setEndDate(e.target.value)}
                        className="w-full bg-[#131b2d] border border-[#FF2A6D]/50 px-4 py-3 text-[#EAEAEA] font-mono focus:border-[#FF2A6D] focus:shadow-[0_0_10px_rgba(255,42,109,0.2)] outline-none transition-all"
                        min={startDate}
                        max={formatDate(oneWeekAhead)}
                    />
                </div>

                <button
                    type="submit"
                    disabled={loading}
                    className="w-full bg-[#00FFF6]/10 hover:bg-[#00FFF6]/20 text-[#00FFF6] border border-[#00FFF6]/50 hover:border-[#00FFF6] hover:shadow-[0_0_15px_rgba(0,255,246,0.3)] font-mono font-bold py-3 px-6 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 transition-all uppercase tracking-widest mt-8"
                >
                    {loading ? 'PROCESSING>>' : (
                        <>
                            INIT_FORECAST <ArrowRight className="w-4 h-4" />
                        </>
                    )}
                </button>
            </form>
        </div>
    );
};

export default ControlPanel;
