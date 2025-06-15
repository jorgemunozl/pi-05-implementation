
import React from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Atom, TestTube, Microscope, FlaskConical, ArrowRight } from 'lucide-react';

const Hero = () => {
  return (
    <section id="home" className="bg-gradient-to-br from-black via-gray-900 to-orange-900 py-20 min-h-screen flex items-center">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid lg:grid-cols-2 gap-12 items-center">
          <div>
            <div className="flex items-center space-x-2 mb-6">
              <div className="w-2 h-2 bg-orange-500 rounded-full"></div>
              <span className="text-orange-500 font-medium uppercase tracking-wide text-sm">Purpose</span>
            </div>
            
            <h1 className="text-5xl md:text-7xl font-bold text-white mb-6 leading-tight">
              Science: Where Knowledge
              <span className="text-transparent bg-gradient-to-r from-orange-500 to-red-500 bg-clip-text block">
                Meets Discovery
              </span>
            </h1>
            
            <p className="text-xl text-gray-300 max-w-2xl mb-8 leading-relaxed">
              The scientific companion that learns and adapts alongside you. Explore groundbreaking research and dive deep into the universe of knowledge.
            </p>
            
            <div className="flex flex-col sm:flex-row gap-4 mb-16">
              <Button size="lg" className="bg-gradient-to-r from-orange-500 to-red-500 hover:from-orange-600 hover:to-red-600 text-white border-0 group">
                Start Exploring
                <ArrowRight className="ml-2 w-4 h-4 group-hover:translate-x-1 transition-transform" />
              </Button>
              <Button size="lg" variant="outline" className="border-gray-600 text-gray-300 hover:bg-gray-800 hover:text-white">
                Watch Videos
              </Button>
            </div>
          </div>
          
          <div className="relative">
            <div className="absolute inset-0 bg-gradient-to-r from-orange-500/20 to-red-500/20 rounded-3xl blur-3xl"></div>
            <div className="relative bg-gradient-to-br from-gray-900 to-black rounded-3xl p-8 border border-gray-800">
              <div className="grid grid-cols-2 gap-6">
                <Card className="bg-gray-800/50 backdrop-blur-sm border-gray-700 hover:border-orange-500/50 transition-all">
                  <CardContent className="p-6 text-center">
                    <Atom className="w-12 h-12 text-orange-500 mx-auto mb-4" />
                    <h3 className="text-lg font-semibold mb-2 text-white">Physics</h3>
                    <p className="text-gray-400 text-sm">Fundamental forces</p>
                  </CardContent>
                </Card>
                
                <Card className="bg-gray-800/50 backdrop-blur-sm border-gray-700 hover:border-orange-500/50 transition-all">
                  <CardContent className="p-6 text-center">
                    <FlaskConical className="w-12 h-12 text-green-500 mx-auto mb-4" />
                    <h3 className="text-lg font-semibold mb-2 text-white">Chemistry</h3>
                    <p className="text-gray-400 text-sm">Molecular interactions</p>
                  </CardContent>
                </Card>
                
                <Card className="bg-gray-800/50 backdrop-blur-sm border-gray-700 hover:border-orange-500/50 transition-all">
                  <CardContent className="p-6 text-center">
                    <Microscope className="w-12 h-12 text-purple-500 mx-auto mb-4" />
                    <h3 className="text-lg font-semibold mb-2 text-white">Biology</h3>
                    <p className="text-gray-400 text-sm">Living systems</p>
                  </CardContent>
                </Card>
                
                <Card className="bg-gray-800/50 backdrop-blur-sm border-gray-700 hover:border-orange-500/50 transition-all">
                  <CardContent className="p-6 text-center">
                    <TestTube className="w-12 h-12 text-red-500 mx-auto mb-4" />
                    <h3 className="text-lg font-semibold mb-2 text-white">Research</h3>
                    <p className="text-gray-400 text-sm">Cutting-edge</p>
                  </CardContent>
                </Card>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Hero;
