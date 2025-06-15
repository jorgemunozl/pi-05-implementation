
import React from 'react';
import { Button } from '@/components/ui/button';
import { Microscope } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

const Navigation = () => {
  const navigate = useNavigate();

  const handleLogoClick = () => {
    navigate('/');
  };

  const handleDiscoveriesClick = () => {
    navigate('/timeline');
  };

  const handleNotesClick = () => {
    navigate('/notes');
  };

  const handleContactClick = () => {
    navigate('/contact');
  };

  return (
    <nav className="bg-black/95 backdrop-blur-sm border-b border-gray-800 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          <div 
            className="flex items-center space-x-2 cursor-pointer hover:opacity-80 transition-opacity"
            onClick={handleLogoClick}
          >
            <Microscope className="w-8 h-8 text-orange-500" />
            <span className="text-xl font-bold text-white">ScienceHub</span>
          </div>
          
          <div className="hidden md:flex items-center space-x-8">
            <button 
              onClick={handleDiscoveriesClick}
              className="text-gray-300 hover:text-orange-500 transition-colors"
            >
              Discoveries
            </button>
            <a href="#latex" className="text-gray-300 hover:text-orange-500 transition-colors">LaTeX</a>
            <button 
              onClick={handleNotesClick}
              className="text-gray-300 hover:text-orange-500 transition-colors"
            >
              Notes
            </button>
            <button 
              onClick={handleContactClick}
              className="text-gray-300 hover:text-orange-500 transition-colors"
            >
              Contact
            </button>
          </div>
          
          <Button className="bg-gradient-to-r from-orange-500 to-red-500 hover:from-orange-600 hover:to-red-600 text-white border-0">
            Get Started
          </Button>
        </div>
      </div>
    </nav>
  );
};

export default Navigation;
