
import React from 'react';

const Index = () => {
  return (
    <div className="min-h-screen flex items-center justify-center bg-background">
      <div className="max-w-lg w-full px-6 py-16 bg-card border rounded shadow text-center">
        <h1 className="text-3xl font-bold mb-6 bg-gradient-to-r from-primary to-purple-600 bg-clip-text text-transparent">
          Welcome to your new project!
        </h1>
        <p className="text-muted-foreground text-lg mb-8">
          This page has been cleared. Start building your app here. 🚀
        </p>
        <div className="opacity-60 text-sm">
          Need inspiration? Add your first button, heading, or feature using the chat!
        </div>
      </div>
    </div>
  );
};

export default Index;
