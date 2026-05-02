import React, { createContext, useContext, useState } from 'react';

const DataContext = createContext(null);

export const DataProvider = ({ children }) => {
  const [data, setData] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [realtimeStatus, setRealtimeStatus] = useState(null);

  return (
    <DataContext.Provider value={{ data, setData, metrics, setMetrics, realtimeStatus, setRealtimeStatus }}>
      {children}
    </DataContext.Provider>
  );
};

export const useData = () => {
  return useContext(DataContext);
};

export default DataContext;
