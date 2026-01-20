import { useState, useEffect } from 'react';
import { Component, Network, Cpu, BarChart3, Target, Database, Layers } from 'lucide-react';
import { getModelMetrics } from '../services/api';

const AboutModels = () => {
    const [metrics, setMetrics] = useState(null);
    
    useEffect(() => {
        const fetchMetrics = async () => {
            try {
                const data = await getModelMetrics();
                setMetrics(data);
            } catch (err) {
                console.error("Failed to fetch metrics");
            }
        };
        fetchMetrics();
    }, []);

    const models = [
        {
            title: 'LightGBM + LSTM Hybrid',
            description: 'A powerful hybrid combining Gradient Boosting (LightGBM) with Long Short-Term Memory neural networks. Captures both tabular patterns and temporal dependencies.',
            features: [
                'LightGBM: 1000 trees for calendar & lag features',
                'LSTM: 2 layers x 64 units for residual patterns',
                'RMSE: 25 MW | MAPE: 0.36%',
                'Best for: Smooth, stable multi-day forecasts'
            ],
            icon: Network,
            color: 'text-[#00FFF6]',
            bg: 'bg-[#00FFF6]/10',
            border: 'border-[#00FFF6]/30'
        },
        {
            title: 'LightGBM + GRU Hybrid',
            description: 'A hybrid model using Gradient Boosting with Gated Recurrent Units (GRU). Faster response to recent changes with efficient memory usage.',
            features: [
                'GRU: Faster training than LSTM ',
                'Better at short-term fluctuations',
                'RMSE: 55 MW | MAPE: 0.53%',
                'Best for: Real-time responsive forecasting'
            ],
            icon: Cpu,
            color: 'text-[#FF2A6D]',
            bg: 'bg-[#FF2A6D]/10',
            border: 'border-[#FF2A6D]/30'
        }
    ];

    return (
        <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-700 font-mono">
            <header className="border-b border-[#00FFF6]/30 pb-6">
                <h2 className="text-3xl font-bold text-[#00FFF6] mb-2 uppercase tracking-wider">Model_Documentation</h2>
                <p className="text-slate-400 text-sm">// HYBRID_FORECASTING_SYSTEM_v2.0</p>
            </header>

            {/* Model Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {models.map((model) => {
                    const Icon = model.icon;
                    return (
                        <div key={model.title} className={`p-6 border ${model.border} ${model.bg} relative overflow-hidden`}>
                            <div className="absolute top-0 left-0 w-2 h-2 border-t border-l border-current" style={{ color: model.color.includes('00FFF6') ? '#00FFF6' : '#FF2A6D' }} />
                            <div className="absolute top-0 right-0 w-2 h-2 border-t border-r border-current" style={{ color: model.color.includes('00FFF6') ? '#00FFF6' : '#FF2A6D' }} />
                            <div className="absolute bottom-0 left-0 w-2 h-2 border-b border-l border-current" style={{ color: model.color.includes('00FFF6') ? '#00FFF6' : '#FF2A6D' }} />
                            <div className="absolute bottom-0 right-0 w-2 h-2 border-b border-r border-current" style={{ color: model.color.includes('00FFF6') ? '#00FFF6' : '#FF2A6D' }} />
                            
                            <div className="flex items-center gap-4 mb-4">
                                <div className={`p-3 bg-[#131b2d] border border-[#1e293b] ${model.color}`}>
                                    <Icon className="w-6 h-6" />
                                </div>
                                <h3 className={`text-lg font-bold ${model.color} uppercase tracking-wider`}>{model.title}</h3>
                            </div>

                            <p className="text-slate-300 mb-6 text-sm leading-relaxed">
                                {model.description}
                            </p>

                            <ul className="space-y-3">
                                {model.features.map((feature, idx) => (
                                    <li key={idx} className="flex items-center gap-2 text-slate-400 text-xs">
                                        <div className={`w-1.5 h-1.5 ${model.color.replace('text-', 'bg-')}`} />
                                        {feature}
                                    </li>
                                ))}
                            </ul>
                        </div>
                    );
                })}
            </div>

            {/* Training Details */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="p-6 bg-[#0B0F1A] border border-[#1e293b]">
                    <div className="flex items-center gap-2 mb-4 text-[#00FFF6]">
                        <Database className="w-5 h-5" />
                        <h4 className="uppercase text-sm tracking-wider">Training_Data</h4>
                    </div>
                    <div className="space-y-2 text-sm">
                        <p className="text-slate-400">Source: <span className="text-white">Delhi SLDC</span></p>
                        <p className="text-slate-400">Period: <span className="text-white">Oct 2025 - Jan 2026</span></p>
                        <p className="text-slate-400">Samples: <span className="text-white">34,560 (5-min intervals)</span></p>
                        <p className="text-slate-400">Features: <span className="text-white">34 engineered features</span></p>
                    </div>
                </div>
                
                <div className="p-6 bg-[#0B0F1A] border border-[#1e293b]">
                    <div className="flex items-center gap-2 mb-4 text-[#FF2A6D]">
                        <Layers className="w-5 h-5" />
                        <h4 className="uppercase text-sm tracking-wider">Feature_Engineering</h4>
                    </div>
                    <div className="space-y-2 text-sm text-slate-400">
                        <p>• Time features (hour, day, month cycles)</p>
                        <p>• Cyclical encoding (sin/cos transforms)</p>
                        <p>• Lag features (1, 2, 3, 6, 12, 24, 288)</p>
                        <p>• Rolling stats (mean, std, min, max)</p>
                    </div>
                </div>
                
                <div className="p-6 bg-[#0B0F1A] border border-[#1e293b]">
                    <div className="flex items-center gap-2 mb-4 text-yellow-400">
                        <Target className="w-5 h-5" />
                        <h4 className="uppercase text-sm tracking-wider">Model_Performance</h4>
                    </div>
                    <div className="space-y-2 text-sm">
                        <p className="text-slate-400">LSTM RMSE: <span className="text-[#00FFF6]">25 MW</span></p>
                        <p className="text-slate-400">GRU RMSE: <span className="text-[#FF2A6D]">55 MW</span></p>
                        <p className="text-slate-400">LSTM MAPE: <span className="text-[#00FFF6]">0.36%</span></p>
                        <p className="text-slate-400">GRU MAPE: <span className="text-[#FF2A6D]">0.53%</span></p>
                    </div>
                </div>
            </div>

            {/* Architecture */}
            <div className="p-8 bg-[#0B0F1A] border border-[#1e293b]">
                <h3 className="text-lg font-bold mb-4 flex items-center gap-2 text-[#00FFF6] uppercase tracking-wider">
                    <Component className="w-5 h-5" />
                    System_Architecture
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                    <div className="space-y-4 text-slate-400 text-sm leading-relaxed">
                        <h4 className="text-white font-bold">Backend (FastAPI + Python)</h4>
                        <ul className="space-y-1">
                            <li>• LightGBM for tabular feature prediction</li>
                            <li>• TensorFlow/Keras LSTM & GRU models</li>
                            <li>• Real-time prediction API endpoints</li>
                            <li>• Continuous learning data pipeline</li>
                        </ul>
                    </div>
                    <div className="space-y-4 text-slate-400 text-sm leading-relaxed">
                        <h4 className="text-white font-bold">Frontend (React + Vite)</h4>
                        <ul className="space-y-1">
                            <li>• Interactive forecast visualization</li>
                            <li>• Real-time grid status monitoring</li>
                            <li>• Model comparison analytics</li>
                            <li>• Responsive cyberpunk UI design</li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default AboutModels;
