const StatsCard = ({ title, value, icon: Icon, color }) => {
    return (
        <div className="bg-[#0B0F1A] border border-[#1e293b] p-4 flex items-center space-x-4 hover:border-[#00FFF6]/50 hover:shadow-[0_0_15px_rgba(0,255,246,0.1)] transition-all group relative overflow-hidden">
            <div className={`p-3 bg-[#131b2d] border border-[#1e293b] group-hover:border-[#00FFF6]/50 transition-colors ${color}`}>
                <Icon className="w-6 h-6" />
            </div>
            <div>
                <p className="text-xs text-[#8A8F98] font-mono uppercase tracking-widest mb-1">{title}</p>
                <p className="text-xl font-bold text-[#EAEAEA] font-mono tracking-tighter">{value}</p>
            </div>
            {/* Tech Deco */}
            <div className="absolute top-0 right-0 w-8 h-8 border-t-2 border-r-2 border-[#1e293b] group-hover:border-[#00FFF6]/30 transition-colors" />
        </div>
    );
};

export default StatsCard;
