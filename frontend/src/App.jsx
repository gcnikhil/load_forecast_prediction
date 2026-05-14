import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import MainLayout from './layouts/MainLayout';
import Dashboard from './pages/Dashboard';
import AboutModels from './pages/AboutModels';
import Analytics from './pages/Analytics';
import WhatIf from './pages/WhatIf';

function App() {
  return (
    <BrowserRouter>
      <MainLayout>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/about" element={<AboutModels />} />
          <Route path="/analytics" element={<Analytics />} />
          <Route path="/whatif" element={<WhatIf />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </MainLayout>
    </BrowserRouter>
  );
}

export default App;
