import axios from 'axios';

// Fix 24: Use VITE_API_URL env variable with fallback for deployment flexibility
const API_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8002';

export const predictLoad = async (startDate, endDate) => {
    try {
        const response = await axios.post(`${API_URL}/predict`, {
            start_date: startDate,
            end_date: endDate
        });
        
        // Map new field names for backward compatibility with UI
        const data = response.data;
        return {
            ...data,
            // Provide both new and legacy field names
            loads_lightgbm: data.loads_lightgbm_gru,
            loads_bengaluru: data.loads_lightgbm_gru,
            loads_lightgbm_lstm: data.loads_lightgbm_gru,
            loads_lightgbm_gru: data.loads_lightgbm_gru
        };
    } catch (error) {
        console.error("Error predicting load:", error);
        throw error;
    }
};

export const checkHealth = async () => {
    try {
        const response = await axios.get(`${API_URL}/health`);
        return response.data;
    } catch (error) {
        return { status: "error", error };
    }
};

export const getModelMetrics = async () => {
    try {
        const response = await axios.get(`${API_URL}/model-metrics`);
        return response.data;
    } catch (error) {
        console.error("Error fetching model metrics:", error);
        throw error;
    }
};

export const getRealtimeStatus = async () => {
    try {
        const response = await axios.get(`${API_URL}/realtime-status`);
        return response.data;
    } catch (error) {
        console.error("Error fetching realtime status:", error);
        throw error;
    }
};

export const getHistoricalAccuracy = async () => {
    try {
        const response = await axios.get(`${API_URL}/historical-accuracy`);
        return response.data;
    } catch (error) {
        console.error("Error fetching historical accuracy:", error);
        throw error;
    }
};


/**
 * What-If prediction — calls the new /whatif/predict endpoint.
 * featureOverrides: plain object, e.g. { is_holiday: 1, temperature_celsius: 38 }
 */
export const whatIfPredict = async (startDate, endDate, featureOverrides) => {
    try {
        const response = await axios.post(`${API_URL}/whatif/predict`, {
            start_date: startDate,
            end_date: endDate,
            feature_overrides: featureOverrides,
        });
        return response.data;
    } catch (error) {
        console.error("What-If prediction error:", error);
        throw error;
    }
};
