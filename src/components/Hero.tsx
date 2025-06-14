
import React from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Atom, TestTube, Microscope, FlaskConical } from 'lucide-react';

const Hero = () => {
  return (
    <section id="home" className="bg-gradient-to-br from-blue-50 to-indigo-100 py-20">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center">
          <h1 className="text-4xl md:text-6xl font-bold text-gray-900 mb-6">
            Explore the Wonders of
            <span className="text-blue-600 block">Science</span>
          </h1>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto mb-8">
            Discover groundbreaking research, explore fascinating experiments, and dive deep into the world of scientific innovation. Join us on a journey through the universe of knowledge.
          </p>
          
          <div className="flex flex-col sm:flex-row gap-4 justify-center mb-16">
            <Button size="lg" className="bg-blue-600 hover:bg-blue-700">
              Start Exploring
            </Button>
            <Button size="lg" variant="outline">
              Watch Videos
            </Button>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mt-16">
            <Card className="bg-white/80 backdrop-blur-sm hover:shadow-lg transition-shadow">
              <CardContent className="p-6 text-center">
                <Atom className="w-12 h-12 text-blue-600 mx-auto mb-4" />
                <h3 className="text-lg font-semibold mb-2">Physics</h3>
                <p className="text-gray-600">Understand the fundamental forces that govern our universe</p>
              </CardContent>
            </Card>
            
            <Card className="bg-white/80 backdrop-blur-sm hover:shadow-lg transition-shadow">
              <CardContent className="p-6 text-center">
                <FlaskConical className="w-12 h-12 text-green-600 mx-auto mb-4" />
                <h3 className="text-lg font-semibold mb-2">Chemistry</h3>
                <p className="text-gray-600">Explore molecular interactions and chemical reactions</p>
              </CardContent>
            </Card>
            
            <Card className="bg-white/80 backdrop-blur-sm hover:shadow-lg transition-shadow">
              <CardContent className="p-6 text-center">
                <Microscope className="w-12 h-12 text-purple-600 mx-auto mb-4" />
                <h3 className="text-lg font-semibold mb-2">Biology</h3>
                <p className="text-gray-600">Discover the complexity and beauty of living systems</p>
              </CardContent>
            </Card>
            
            <Card className="bg-white/80 backdrop-blur-sm hover:shadow-lg transition-shadow">
              <CardContent className="p-6 text-center">
                <TestTube className="w-12 h-12 text-red-600 mx-auto mb-4" />
                <h3 className="text-lg font-semibold mb-2">Research</h3>
                <p className="text-gray-600">Access cutting-edge research and scientific papers</p>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Hero;
