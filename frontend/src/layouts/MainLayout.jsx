import Navigation from '../components/Navigation';

const MainLayout = ({ children }) => {
    return (
        <div className="min-h-screen bg-slate-950 text-slate-100 font-inter selection:bg-cyan-500/30">
            <Navigation />
            <main className="ml-64 min-h-screen overflow-x-hidden">
                <div className="p-4 md:p-6 lg:p-8 w-full">
                    {children}
                </div>
            </main>
        </div>
    );
};

export default MainLayout;
