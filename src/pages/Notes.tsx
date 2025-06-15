
import React, { useState } from 'react';
import Navigation from '@/components/Navigation';
import Footer from '@/components/Footer';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';
import { BookOpen, Calendar, Trash2 } from 'lucide-react';

interface Note {
  id: string;
  title: string;
  content: string;
  date: string;
}

const Notes = () => {
  const [notes, setNotes] = useState<Note[]>([]);
  const [title, setTitle] = useState('');
  const [content, setContent] = useState('');

  const addNote = () => {
    if (title.trim() && content.trim()) {
      const newNote: Note = {
        id: Date.now().toString(),
        title: title.trim(),
        content: content.trim(),
        date: new Date().toLocaleDateString()
      };
      setNotes([newNote, ...notes]);
      setTitle('');
      setContent('');
    }
  };

  const deleteNote = (id: string) => {
    setNotes(notes.filter(note => note.id !== id));
  };

  return (
    <div className="min-h-screen bg-black">
      <Navigation />
      
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="text-center mb-12">
          <div className="flex items-center justify-center space-x-2 mb-4">
            <div className="w-2 h-2 bg-orange-500 rounded-full"></div>
            <span className="text-orange-500 font-medium uppercase tracking-wide text-sm">Knowledge</span>
          </div>
          <div className="flex items-center justify-center space-x-2 mb-4">
            <BookOpen className="w-8 h-8 text-orange-500" />
            <h1 className="text-4xl font-bold text-white">Physics Notes</h1>
          </div>
          <p className="text-xl text-gray-400">
            Capture your physics insights, formulas, and discoveries
          </p>
        </div>

        <div className="grid lg:grid-cols-2 gap-8">
          {/* Add Note Form */}
          <div>
            <Card className="bg-gray-900 border-gray-800">
              <CardHeader>
                <CardTitle className="text-white">Add New Note</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <Label htmlFor="title" className="text-gray-300">Title</Label>
                  <Input 
                    id="title"
                    placeholder="Enter note title..."
                    value={title}
                    onChange={(e) => setTitle(e.target.value)}
                    className="bg-gray-800 border-gray-700 text-white placeholder:text-gray-400 focus:border-orange-500"
                  />
                </div>
                <div>
                  <Label htmlFor="content" className="text-gray-300">Content</Label>
                  <Textarea 
                    id="content"
                    placeholder="Write your physics note here... You can include formulas, observations, questions, etc."
                    value={content}
                    onChange={(e) => setContent(e.target.value)}
                    className="min-h-[200px] bg-gray-800 border-gray-700 text-white placeholder:text-gray-400 focus:border-orange-500"
                  />
                </div>
                <Button 
                  onClick={addNote} 
                  className="w-full bg-gradient-to-r from-orange-500 to-red-500 hover:from-orange-600 hover:to-red-600 text-white border-0"
                  disabled={!title.trim() || !content.trim()}
                >
                  Add Note
                </Button>
              </CardContent>
            </Card>
          </div>

          {/* Notes List */}
          <div>
            <div className="mb-4">
              <h2 className="text-2xl font-semibold text-white">
                Your Notes ({notes.length})
              </h2>
            </div>
            
            <div className="space-y-4 max-h-[600px] overflow-y-auto">
              {notes.length === 0 ? (
                <Card className="bg-gray-900 border-gray-800">
                  <CardContent className="text-center py-12">
                    <BookOpen className="w-12 h-12 text-gray-600 mx-auto mb-4" />
                    <p className="text-gray-500">No notes yet. Start writing your first physics note!</p>
                  </CardContent>
                </Card>
              ) : (
                notes.map((note) => (
                  <Card key={note.id} className="bg-gray-900 border-gray-800 hover:border-orange-500/50 transition-all">
                    <CardHeader className="pb-2">
                      <div className="flex justify-between items-start">
                        <CardTitle className="text-lg text-white">{note.title}</CardTitle>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => deleteNote(note.id)}
                          className="text-red-400 hover:text-red-300 hover:bg-red-900/20"
                        >
                          <Trash2 className="w-4 h-4" />
                        </Button>
                      </div>
                      <div className="flex items-center text-sm text-gray-500">
                        <Calendar className="w-4 h-4 mr-1" />
                        {note.date}
                      </div>
                    </CardHeader>
                    <CardContent>
                      <p className="text-gray-300 whitespace-pre-wrap">{note.content}</p>
                    </CardContent>
                  </Card>
                ))
              )}
            </div>
          </div>
        </div>
      </div>

      <Footer />
    </div>
  );
};

export default Notes;
