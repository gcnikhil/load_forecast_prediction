import { Link, useLocation } from 'react-router-dom';
import { LayoutDashboard, Info, BarChart2, Zap } from 'lucide-react';
import clsx from 'clsx';

const Navigation = () => {
    const location = useLocation();

    const navItems = [
        { path: '/', label: 'Dashboard', icon: LayoutDashboard },
        { path: '/analytics', label: 'Analytics', icon: BarChart2 },
        { path: '/about', label: 'About Models', icon: Info },
    ];

    return (
        <div className="h-screen w-64 bg-[#0B0F1A] border-r border-[#00FFF6]/20 flex flex-col fixed left-0 top-0 z-50 transition-all duration-300">
            <div className="p-8 flex items-center gap-3">
                <div className="p-2 border border-[#00FFF6]/50 bg-[#00FFF6]/10 shadow-[0_0_15px_rgba(0,255,246,0.2)]">
                    <Zap className="w-6 h-6 text-[#00FFF6]" />
                </div>
                <h1 className="text-xl font-bold text-[#00FFF6] font-mono tracking-wider">
                    GRID_FCST
                </h1>
            </div>

            <nav className="flex-1 px-4 py-6 space-y-2">
                {navItems.map((item) => {
                    const Icon = item.icon;
                    const isActive = location.pathname === item.path;

                    return (
                        <Link
                            key={item.path}
                            to={item.path}
                            className={clsx(
                                'flex items-center gap-3 px-4 py-3 border transition-all duration-200 group font-mono text-sm relative overflow-hidden hover-glitch',
                                isActive
                                    ? 'bg-[#00FFF6]/10 border-[#00FFF6] text-[#00FFF6] shadow-[0_0_10px_rgba(0,255,246,0.15)]'
                                    : 'border-transparent text-[#8A8F98] hover:text-[#00FFF6] hover:border-[#00FFF6]/30 hover:bg-[#00FFF6]/5'
                            )}
                        >
                            <Icon className={clsx(
                                'w-5 h-5 transition-colors',
                                isActive ? 'text-[#00FFF6]' : 'text-[#8A8F98] group-hover:text-[#00FFF6]'
                            )} />
                            <span className="font-medium tracking-tight">{item.label.toUpperCase()}</span>
                            {isActive && (
                                <div className="absolute left-0 top-0 bottom-0 w-1 bg-[#00FFF6] shadow-[0_0_8px_#00FFF6]" />
                            )}
                        </Link>
                    );
                })}
            </nav>

            <div className="p-4 m-4 border border-[#F9F871]/30 bg-[#F9F871]/5">
                <div className="flex items-center gap-2 mb-2">
                    <div className="w-2 h-2 bg-[#F9F871] shadow-[0_0_8px_#F9F871] animate-pulse" />
                    <span className="text-xs font-mono text-[#F9F871]">SYSTEM::ONLINE</span>
                </div>
                <p className="text-[10px] uppercase tracking-widest text-[#8A8F98] font-mono">
                    Ver 1.0.0
                </p>
            </div>
        </div>
    );
};

export default Navigation;
