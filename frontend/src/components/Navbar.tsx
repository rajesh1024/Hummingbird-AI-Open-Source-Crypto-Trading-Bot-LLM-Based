
import { useState } from 'react';
import { Link } from 'react-router-dom';
import { MenuIcon, XIcon } from 'lucide-react';
import { Button } from "@/components/ui/button";

const Navbar = () => {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <nav className="bg-hummingbird-blue-800 dark:bg-slate-900 px-4 py-2.5 w-full z-20 top-0 left-0 border-b border-hummingbird-blue-700 dark:border-slate-700">
      <div className="container flex flex-wrap justify-between items-center mx-auto">
        <Link to="/" className="flex items-center">
          <span className="text-xl font-semibold text-white">
            <span className="text-hummingbird-teal-400">Hummingbird</span> AI
          </span>
        </Link>
        
        <div className="flex md:order-2">
          
          <button
            type="button"
            onClick={() => setIsOpen(!isOpen)}
            className="inline-flex items-center p-2 ml-3 text-sm text-gray-200 rounded-lg md:hidden hover:bg-hummingbird-blue-700 dark:hover:bg-slate-800 focus:outline-none focus:ring-2 focus:ring-hummingbird-teal-400"
          >
            {isOpen ? <XIcon className="w-6 h-6" /> : <MenuIcon className="w-6 h-6" />}
          </button>
        </div>
        
        <div className={`${isOpen ? 'block' : 'hidden'} justify-between items-center w-full md:flex md:w-auto md:order-1`}>
          <ul className="flex flex-col p-4 mt-4 bg-hummingbird-blue-800 dark:bg-slate-900 rounded-lg border border-hummingbird-blue-700 dark:border-slate-700 md:flex-row md:space-x-8 md:mt-0 md:text-sm md:font-medium md:border-0">
            <li>
              <Link to="/" className="block py-2 pr-4 pl-3 text-white rounded hover:bg-hummingbird-blue-700 dark:hover:bg-slate-800 md:hover:bg-transparent md:hover:text-hummingbird-teal-400 md:p-0">
                Dashboard
              </Link>
            </li>
            <li>
              <Link to="/history" className="block py-2 pr-4 pl-3 text-white rounded hover:bg-hummingbird-blue-700 dark:hover:bg-slate-800 md:hover:bg-transparent md:hover:text-hummingbird-teal-400 md:p-0">
                Position History
              </Link>
            </li>
          </ul>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
