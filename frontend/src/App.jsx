import React from 'react';
import { Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';

// Pages
import PredictPage from './pages/PredictPage';
import StudiesPage from './pages/StudiesPage';
import StudyViewerPage from './pages/StudyViewerPage';
import ReportPage from './pages/ReportPage';
import FairnessPage from './pages/FairnessPage';
import DriftPage from './pages/DriftPage';
import AuditPage from './pages/AuditPage';
import SettingsPage from './pages/SettingsPage';

function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<PredictPage />} />
        <Route path="/studies" element={<StudiesPage />} />
        <Route path="/studies/:studyUid" element={<StudyViewerPage />} />
        <Route path="/reports/:studyUid" element={<ReportPage />} />
        <Route path="/fairness" element={<FairnessPage />} />
        <Route path="/drift" element={<DriftPage />} />
        <Route path="/audit" element={<AuditPage />} />
        <Route path="/settings" element={<SettingsPage />} />
      </Routes>
    </Layout>
  );
}

export default App;
