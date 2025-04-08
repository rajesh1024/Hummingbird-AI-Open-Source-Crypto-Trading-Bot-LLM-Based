import { Outlet } from 'react-router-dom';
import Navbar from './Navbar';
import useWebSocketData from '@/hooks/useWebSocketData';
import { BadgeCheck, WifiOff } from 'lucide-react';
import ThemeToggle from './ThemeToggle';

// WebSocket URL configuration
const WS_URL = import.meta.env.VITE_WS_URL + "/ws/dashboard";

const Layout = () => {
  const { isConnected } = useWebSocketData(WS_URL);

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-900 dark:text-slate-100 transition-colors duration-200">
      <Navbar />
      <div className="flex items-center justify-between px-4 py-1 bg-slate-100 dark:bg-slate-800 border-b dark:border-slate-700">
        <div className="ml-auto">
          <ThemeToggle />
        </div>
        {/* <div className="flex items-center text-sm ml-4">
          <span className="mr-2">Data Feed:</span>
          {isConnected ? (
            <div className="flex items-center text-green-600 dark:text-green-500">
              <BadgeCheck size={16} className="mr-1" />
              <span>Connected</span>
            </div>
          ) : (
            <div className="flex items-center text-red-600 dark:text-red-500">
              <WifiOff size={16} className="mr-1" />
              <span>Disconnected</span>
            </div>
          )}
        </div> */}
      </div>
      <main className="container mx-auto py-6 px-4">
        <Outlet />
      </main>
    </div>
  );
};

export default Layout;
