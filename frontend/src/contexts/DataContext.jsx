import React, { createContext, useContext, useState } from 'react';

const DataContext = createContext(null);

export const DataProvider = ({ children }) => {
  const [data, setData] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [realtimeStatus, setRealtimeStatus] = useState(null);
  // Fix 25: Shared loading and error state so pages don't manage redundant
  // local copies — prevents inconsistent UI when switching pages mid-request
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  return (
    <DataContext.Provider value={{
      data, setData,
      metrics, setMetrics,
      realtimeStatus, setRealtimeStatus,
      loading, setLoading,
      error, setError,
    }}>
      {children}
    </DataContext.Provider>
  );
};

// eslint-disable-next-line react-refresh/only-export-components
export const useData = () => {
  return useContext(DataContext);
};

export default DataContext;
