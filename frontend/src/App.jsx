import { Suspense, lazy, useEffect } from 'react';
import { Navigate, Outlet, Route, Routes, useLocation } from 'react-router-dom';

import AppShell from './components/layout/AppShell';
import AuthLayout from './components/layout/AuthLayout';
import { PageFallback } from './components/layout/PageFallback';
import { CatalogProvider } from './state/CatalogContext';
import { SessionProvider } from './state/SessionContext';
import { sessionStore } from './state/sessionStore';
import { useToast } from './state/ToastContext';

const TrainPage = lazy(() => import('./pages/TrainPage'));
const DatasetsPage = lazy(() => import('./pages/DatasetsPage'));
const PlaygroundPage = lazy(() => import('./pages/PlaygroundPage'));
const ModelsPage = lazy(() => import('./pages/ModelsPage'));
const FunctionsPage = lazy(() => import('./pages/FunctionsPage'));
const LearnPage = lazy(() => import('./pages/LearnPage'));
const LoginPage = lazy(() => import('./pages/LoginPage'));
const SignupPage = lazy(() => import('./pages/SignupPage'));

/**
 * Mounts the catalogues + training session only for authenticated pages, so
 * the login screen never fires API calls that would 401 and redirect.
 */
function WorkspaceProviders() {
  return (
    <CatalogProvider>
      <SessionProvider>
        <Outlet />
      </SessionProvider>
    </CatalogProvider>
  );
}

export default function App() {
  const toast = useToast();
  const location = useLocation();

  // Bridge store notifications (training errors, build failures) to toasts.
  useEffect(() => {
    sessionStore.onNotify = (message, tone) => {
      if (tone === 'neg') toast.error(message);
      else if (tone === 'pos') toast.success(message);
      else if (tone === 'warn') toast.warn(message);
      else toast.info(message);
    };
    return () => {
      sessionStore.onNotify = null;
    };
  }, [toast]);

  useEffect(() => {
    document.documentElement.dataset.route = location.pathname.slice(1) || 'train';
  }, [location.pathname]);

  return (
    <Suspense fallback={<PageFallback />}>
      <Routes>
        <Route
          path="/login"
          element={
            <AuthLayout title="Welcome back">
              <LoginPage />
            </AuthLayout>
          }
        />
        <Route
          path="/signup"
          element={
            <AuthLayout title="Create your studio">
              <SignupPage />
            </AuthLayout>
          }
        />

        <Route element={<WorkspaceProviders />}>
          <Route element={<AppShell />}>
            <Route index element={<Navigate to="/train" replace />} />
            <Route path="/train" element={<TrainPage />} />
            <Route path="/datasets" element={<DatasetsPage />} />
            <Route path="/playground" element={<PlaygroundPage />} />
            <Route path="/models" element={<ModelsPage />} />
            <Route path="/functions" element={<FunctionsPage />} />
            <Route path="/learn" element={<LearnPage />} />
            <Route path="*" element={<Navigate to="/train" replace />} />
          </Route>
        </Route>
      </Routes>
    </Suspense>
  );
}
