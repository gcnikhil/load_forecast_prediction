import Navigation from '../components/Navigation';
import { DataProvider } from '../contexts/DataContext';

const MainLayout = ({ children }) => {
    return (
        <DataProvider>
            <div className="min-h-screen bg-slate-950 text-slate-100 font-inter selection:bg-cyan-500/30">
                <Navigation />
                <main className="ml-64 min-h-screen overflow-x-hidden">
                    <div className="p-4 md:p-6 lg:p-8 w-full">
                        {children}
                    </div>
                </main>
            </div>
        </DataProvider>
    );
};

export default MainLayout;
