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

    // Calculate maximum allowed end date (start date + 7 days)
    const maxEndDateObj = new Date(startDate);
    maxEndDateObj.setDate(maxEndDateObj.getDate() + 7);
    const maxEndDate = formatDate(maxEndDateObj);

    const handleStartDateChange = (e) => {
        const newStart = e.target.value;
        setStartDate(newStart);
        
        // Ensure endDate is valid given the new startDate
        const newStartObj = new Date(newStart);
        const currentEndObj = new Date(endDate);
        const newMaxEndObj = new Date(newStart);
        newMaxEndObj.setDate(newMaxEndObj.getDate() + 7);
        
        if (currentEndObj < newStartObj) {
            setEndDate(newStart);
        } else if (currentEndObj > newMaxEndObj) {
            setEndDate(formatDate(newMaxEndObj));
        }
    };

    const handleSubmit = (e) => {
        e.preventDefault();
        onPredict(startDate, endDate);
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
                        onChange={handleStartDateChange}
                        className="w-full bg-[#131b2d] border border-[#00FFF6]/50 px-4 py-3 text-[#EAEAEA] font-mono focus:border-[#00FFF6] focus:shadow-[0_0_10px_rgba(0,255,246,0.2)] outline-none transition-all placeholder-[#8A8F98]"
                    />
                </div>

                <div className="space-y-2">
                    <label className="text-xs font-mono text-[#FF2A6D]/70 flex items-center gap-2 uppercase tracking-wider">
                        <Clock className="w-3 h-3" /> End_Date
                    </label>
                    <input
                        type="date"
                        value={endDate}
                        min={startDate}
                        max={maxEndDate}
                        onChange={(e) => setEndDate(e.target.value)}
                        className="w-full bg-[#131b2d] border border-[#FF2A6D]/50 px-4 py-3 text-[#EAEAEA] font-mono focus:border-[#FF2A6D] focus:shadow-[0_0_10px_rgba(255,42,109,0.2)] outline-none transition-all"
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
