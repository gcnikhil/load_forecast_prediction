import { useEffect, useState } from 'react';
import { Component, Network, Cpu, Target, Database, Layers } from 'lucide-react';
import { getModelMetrics } from '../services/api';

const AboutModels = () => {
    const [metrics, setMetrics] = useState(null);

    useEffect(() => {
        const fetchMetrics = async () => {
            try {
                const data = await getModelMetrics();
                setMetrics(data);
            } catch (error) {
                console.error("Failed to fetch metrics", error);
            }
        };
        fetchMetrics();
    }, []);

    const models = [
        {
            title: 'LightGBM + GRU Hybrid Model (Active Production)',
            description: 'The active production model. It combines the strong tabular forecasting capabilities of LightGBM with a Gated Recurrent Unit (GRU) neural network to capture multi-step temporal dynamics and correct short-term residual errors via stacking.',
            features: [
                `Architecture: ${metrics?.gru_hybrid?.architecture ?? 'LightGBM + GRU stacking ensemble'}`,
                `Features: ${metrics?.gru_hybrid?.features ?? 62} engineered inputs`,
                `RMSE: ${metrics?.gru_hybrid?.rmse_mw ?? '928.52'} MW`,
                `MAPE: ${metrics?.gru_hybrid?.mape_percent ?? '6.59'}%`,
                `MAE: ${metrics?.gru_hybrid?.mae_mw ?? '718.10'} MW`,
                `Bias: +${metrics?.gru_hybrid?.bias_mw ?? '382.33'} MW`,
                'Best for: Highly responsive real-time demand forecasting'
            ],
            icon: Cpu,
            color: 'text-[#00FFF6]',
            bg: 'bg-[#00FFF6]/5',
            border: 'border-[#00FFF6]/30'
        },
        {
            title: 'Base GRU Neural Network (Baseline)',
            description: 'A pure Gated Recurrent Unit (GRU) recurrent neural network with attention mechanisms trained on sequence history. Serves as the temporal pattern predictor, feeding its forecasts as stage-1 inputs into the ensemble.',
            features: [
                'Architecture: Recurrent Neural Network (GRU + Self-Attention)',
                'Features: Sequence window inputs',
                `RMSE: ${metrics?.gru_only?.rmse_mw ?? '1303.45'} MW`,
                `MAPE: ${metrics?.gru_only?.mape_percent ?? '9.25'}%`,
                `MAE: ${metrics?.gru_only?.mae_mw ?? '999.17'} MW`,
                `Bias: +${metrics?.gru_only?.bias_mw ?? '373.08'} MW`,
                'Best for: Learning long-term temporal sequence patterns'
            ],
            icon: Network,
            color: 'text-[#FF2A6D]',
            bg: 'bg-[#FF2A6D]/5',
            border: 'border-[#FF2A6D]/20'
        }
    ];

    return (
        <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-700 font-mono">
            <header className="border-b border-[#00FFF6]/30 pb-6">
                <h2 className="text-3xl font-bold text-[#00FFF6] mb-2 uppercase tracking-wider">Model_Documentation</h2>
                <p className="text-slate-400 text-sm">// ACTIVE_HYBRID_FORECASTING_SYSTEM_v2.5</p>
            </header>

            <div className="grid grid-cols-1 gap-6 max-w-3xl">
                {models.map((model) => {
                    const Icon = model.icon;
                    return (
                        <div key={model.title} className={`p-6 border ${model.border} ${model.bg} relative overflow-hidden`}>
                            <div className="absolute top-0 left-0 w-2 h-2 border-t border-l border-[#FF2A6D]" />
                            <div className="absolute top-0 right-0 w-2 h-2 border-t border-r border-[#FF2A6D]" />
                            <div className="absolute bottom-0 left-0 w-2 h-2 border-b border-l border-[#FF2A6D]" />
                            <div className="absolute bottom-0 right-0 w-2 h-2 border-b border-r border-[#FF2A6D]" />

                            <div className="flex items-center gap-4 mb-4">
                                <div className={`p-3 bg-[#131b2d] border border-[#FF2A6D]/30 ${model.color}`}>
                                    <Icon className="w-6 h-6 animate-pulse" />
                                </div>
                                <h3 className={`text-lg font-bold ${model.color} uppercase tracking-wider`}>{model.title}</h3>
                            </div>

                            <p className="text-slate-300 mb-6 text-sm leading-relaxed">
                                {model.description}
                            </p>

                            <ul className="space-y-3">
                                {model.features.map((feature, idx) => (
                                    <li key={idx} className="flex items-center gap-2 text-slate-400 text-xs">
                                        <div className="w-1.5 h-1.5 bg-[#FF2A6D]" />
                                        {feature}
                                    </li>
                                ))}
                            </ul>
                        </div>
                    );
                })}
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="p-6 bg-[#0B0F1A] border border-[#1e293b]">
                    <div className="flex items-center gap-2 mb-4 text-[#00FFF6]">
                        <Database className="w-5 h-5" />
                        <h4 className="uppercase text-sm tracking-wider">Training_Data</h4>
                    </div>
                    <div className="space-y-2 text-sm">
                        <p className="text-slate-400">Source: <span className="text-white">{metrics?.data_source ?? 'Karnataka SLDC / BESCOM dataset'}</span></p>
                        <p className="text-slate-400">Period: <span className="text-white">{metrics?.training_period ?? 'Based on saved model metadata'}</span></p>
                        <p className="text-slate-400">Samples: <span className="text-white">{metrics?.gru_hybrid?.training_samples ?? '--'} training rows</span></p>
                        <p className="text-slate-400">Features: <span className="text-white">{metrics?.gru_hybrid?.features ?? '--'} engineered features</span></p>
                    </div>
                </div>

                <div className="p-6 bg-[#0B0F1A] border border-[#1e293b]">
                    <div className="flex items-center gap-2 mb-4 text-[#FF2A6D]">
                        <Layers className="w-5 h-5" />
                        <h4 className="uppercase text-sm tracking-wider">Feature_Engineering</h4>
                    </div>
                    <div className="space-y-2 text-sm text-slate-400">
                        <p>- Time features (hour, day, month cycles)</p>
                        <p>- Cyclical encoding (sin/cos transforms)</p>
                        <p>- Lag features (1, 2, 3, 6, 12, 24, 168)</p>
                        <p>- Rolling stats (mean, std, min, max)</p>
                        <p>- Temperature, humidity, holiday flags</p>
                    </div>
                </div>

                <div className="p-6 bg-[#0B0F1A] border border-[#1e293b]">
                    <div className="flex items-center gap-2 mb-4 text-yellow-400">
                        <Target className="w-5 h-5" />
                        <h4 className="uppercase text-sm tracking-wider">Model_Performance</h4>
                    </div>
                    <div className="space-y-2 text-sm">
                        <p className="text-slate-400">Stacking RMSE: <span className="text-[#00FFF6] font-bold">{metrics?.gru_hybrid?.rmse_mw ?? '928.52'} MW</span></p>
                        <p className="text-slate-400">Stacking MAPE: <span className="text-[#00FFF6] font-bold">{metrics?.gru_hybrid?.mape_percent ?? '6.59'}%</span></p>
                        <p className="text-slate-400">Baseline GRU MAPE: <span className="text-[#FF2A6D] font-bold">{metrics?.gru_only?.mape_percent ?? '9.25'}%</span></p>
                        <p className="text-slate-400 font-mono text-[10px] mt-2 text-slate-500">// Stacking reduces error by ~29%</p>
                    </div>
                </div>
            </div>

            <div className="p-8 bg-[#0B0F1A] border border-[#1e293b]">
                <h3 className="text-lg font-bold mb-4 flex items-center gap-2 text-[#00FFF6] uppercase tracking-wider">
                    <Component className="w-5 h-5" />
                    System_Architecture
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                    <div className="space-y-4 text-slate-400 text-sm leading-relaxed">
                        <h4 className="text-white font-bold">Backend (FastAPI + Python)</h4>
                        <ul className="space-y-1">
                            <li>- LightGBM for residual error correction</li>
                            <li>- TensorFlow/Keras GRU temporal residual model</li>
                            <li>- Real-time prediction and what-if simulation APIs</li>
                            <li>- Automated live SLDC scrapers</li>
                        </ul>
                    </div>
                    <div className="space-y-4 text-slate-400 text-sm leading-relaxed">
                        <h4 className="text-white font-bold">Frontend (React + Vite)</h4>
                        <ul className="space-y-1">
                            <li>- Interactive forecast profile visualization</li>
                            <li>- Real-time grid status monitoring</li>
                            <li>- Power system load analysis (LDC, Load Factor)</li>
                            <li>- Cyberpunk cyberpunk UI design</li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default AboutModels;
