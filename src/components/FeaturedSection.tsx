
import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';

const FeaturedSection = () => {
  const discoveries = [
    {
      title: "Quantum Computing Breakthrough",
      category: "Physics",
      description: "Scientists achieve new milestone in quantum error correction, bringing us closer to practical quantum computers.",
      image: "https://images.unsplash.com/photo-1518770660439-4636190af475?w=400&h=250&fit=crop",
      badge: "Latest",
      badgeColor: "bg-orange-500"
    },
    {
      title: "CRISPR Gene Editing Advances",
      category: "Biology",
      description: "New CRISPR techniques show promise for treating genetic diseases with unprecedented precision.",
      image: "https://images.unsplash.com/photo-1535268647677-300dbf3d78d1?w=400&h=250&fit=crop",
      badge: "Trending",
      badgeColor: "bg-green-500"
    },
    {
      title: "Climate Change Research",
      category: "Environmental Science",
      description: "Comprehensive study reveals new insights into global warming patterns and mitigation strategies.",
      image: "https://images.unsplash.com/photo-1470071459604-3b5ec3a7fe05?w=400&h=250&fit=crop",
      badge: "Important",
      badgeColor: "bg-red-500"
    }
  ];

  return (
    <section id="discoveries" className="py-20 bg-gray-900">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
            Latest Scientific Discoveries
          </h2>
          <p className="text-xl text-gray-400 max-w-2xl mx-auto">
            Stay updated with the most recent breakthroughs and discoveries from the scientific community
          </p>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {discoveries.map((discovery, index) => (
            <Card key={index} className="overflow-hidden bg-gray-800 border-gray-700 hover:border-orange-500/50 transition-all hover:shadow-2xl">
              <div className="relative">
                <img 
                  src={discovery.image} 
                  alt={discovery.title}
                  className="w-full h-48 object-cover"
                />
                <Badge className={`absolute top-4 right-4 ${discovery.badgeColor} text-white border-0`}>
                  {discovery.badge}
                </Badge>
              </div>
              <CardHeader>
                <div className="flex items-center justify-between mb-2">
                  <Badge variant="outline" className="border-gray-600 text-gray-300">{discovery.category}</Badge>
                </div>
                <CardTitle className="text-xl text-white">{discovery.title}</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-gray-400 mb-4">{discovery.description}</p>
                <Button variant="outline" className="w-full border-gray-600 text-gray-300 hover:bg-orange-500 hover:text-white hover:border-orange-500">
                  Read More
                </Button>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </section>
  );
};

export default FeaturedSection;
