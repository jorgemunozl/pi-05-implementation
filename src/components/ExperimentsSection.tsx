
import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { TestTube, FlaskConical, Microscope, Atom } from 'lucide-react';

const ExperimentsSection = () => {
  const experiments = [
    {
      title: "Virtual Chemistry Lab",
      description: "Conduct safe chemistry experiments in a virtual environment",
      icon: FlaskConical,
      color: "text-green-600",
      difficulty: "Beginner"
    },
    {
      title: "Physics Simulations",
      description: "Explore physics concepts through interactive simulations",
      icon: Atom,
      color: "text-blue-600",
      difficulty: "Intermediate"
    },
    {
      title: "Microscopy Studies",
      description: "Examine cellular structures and microorganisms",
      icon: Microscope,
      color: "text-purple-600",
      difficulty: "Advanced"
    },
    {
      title: "Laboratory Techniques",
      description: "Learn essential laboratory skills and procedures",
      icon: TestTube,
      color: "text-red-600",
      difficulty: "All Levels"
    }
  ];

  return (
    <section id="experiments" className="py-20 bg-gray-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">
            Interactive Experiments
          </h2>
          <p className="text-xl text-gray-600 max-w-2xl mx-auto">
            Hands-on learning through virtual experiments and simulations
          </p>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {experiments.map((experiment, index) => {
            const IconComponent = experiment.icon;
            return (
              <Card key={index} className="bg-white hover:shadow-lg transition-shadow">
                <CardHeader className="text-center">
                  <IconComponent className={`w-16 h-16 mx-auto mb-4 ${experiment.color}`} />
                  <CardTitle className="text-lg">{experiment.title}</CardTitle>
                </CardHeader>
                <CardContent className="text-center">
                  <p className="text-gray-600 mb-4">{experiment.description}</p>
                  <div className="mb-4">
                    <span className="text-sm bg-gray-100 text-gray-700 px-2 py-1 rounded">
                      {experiment.difficulty}
                    </span>
                  </div>
                  <Button className="w-full">
                    Start Experiment
                  </Button>
                </CardContent>
              </Card>
            );
          })}
        </div>
      </div>
    </section>
  );
};

export default ExperimentsSection;
