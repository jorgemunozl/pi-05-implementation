
import React from 'react';
import { Microscope } from 'lucide-react';

const Footer = () => {
  return (
    <footer className="bg-gray-900 border-t border-gray-800 text-white py-12">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
          <div className="col-span-1 md:col-span-2">
            <div className="flex items-center space-x-2 mb-4">
              <Microscope className="w-8 h-8 text-orange-500" />
              <span className="text-xl font-bold">ScienceHub</span>
            </div>
            <p className="text-gray-400 max-w-md">
              Advancing scientific knowledge through education, research, and innovation. 
              Join our community of curious minds exploring the wonders of science.
            </p>
          </div>
          
          <div>
            <h3 className="text-lg font-semibold mb-4">Quick Links</h3>
            <ul className="space-y-2">
              <li><a href="#home" className="text-gray-400 hover:text-orange-500 transition-colors">Home</a></li>
              <li><a href="#discoveries" className="text-gray-400 hover:text-orange-500 transition-colors">Discoveries</a></li>
              <li><a href="#experiments" className="text-gray-400 hover:text-orange-500 transition-colors">Experiments</a></li>
              <li><a href="#research" className="text-gray-400 hover:text-orange-500 transition-colors">Research</a></li>
            </ul>
          </div>
          
          <div>
            <h3 className="text-lg font-semibold mb-4">Contact</h3>
            <ul className="space-y-2 text-gray-400">
              <li>Email: info@sciencehub.com</li>
              <li>Phone: (555) 123-4567</li>
              <li>Address: 123 Science St, Research City</li>
            </ul>
          </div>
        </div>
        
        <div className="border-t border-gray-800 mt-8 pt-8 text-center">
          <p className="text-gray-400">
            © 2024 ScienceHub. All rights reserved. | Advancing science through education and research.
          </p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
