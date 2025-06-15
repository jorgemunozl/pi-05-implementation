
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
    <div className="min-h-screen bg-white">
      <Navigation />
      
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="text-center mb-12">
          <div className="flex items-center justify-center space-x-2 mb-4">
            <BookOpen className="w-8 h-8 text-blue-600" />
            <h1 className="text-4xl font-bold text-gray-900">Physics Notes</h1>
          </div>
          <p className="text-xl text-gray-600">
            Capture your physics insights, formulas, and discoveries
          </p>
        </div>

        <div className="grid lg:grid-cols-2 gap-8">
          {/* Add Note Form */}
          <div>
            <Card>
              <CardHeader>
                <CardTitle>Add New Note</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <Label htmlFor="title">Title</Label>
                  <Input 
                    id="title"
                    placeholder="Enter note title..."
                    value={title}
                    onChange={(e) => setTitle(e.target.value)}
                  />
                </div>
                <div>
                  <Label htmlFor="content">Content</Label>
                  <Textarea 
                    id="content"
                    placeholder="Write your physics note here... You can include formulas, observations, questions, etc."
                    value={content}
                    onChange={(e) => setContent(e.target.value)}
                    className="min-h-[200px]"
                  />
                </div>
                <Button 
                  onClick={addNote} 
                  className="w-full bg-blue-600 hover:bg-blue-700"
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
              <h2 className="text-2xl font-semibold text-gray-900">
                Your Notes ({notes.length})
              </h2>
            </div>
            
            <div className="space-y-4 max-h-[600px] overflow-y-auto">
              {notes.length === 0 ? (
                <Card>
                  <CardContent className="text-center py-12">
                    <BookOpen className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                    <p className="text-gray-500">No notes yet. Start writing your first physics note!</p>
                  </CardContent>
                </Card>
              ) : (
                notes.map((note) => (
                  <Card key={note.id} className="hover:shadow-md transition-shadow">
                    <CardHeader className="pb-2">
                      <div className="flex justify-between items-start">
                        <CardTitle className="text-lg">{note.title}</CardTitle>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => deleteNote(note.id)}
                          className="text-red-500 hover:text-red-700 hover:bg-red-50"
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
                      <p className="text-gray-700 whitespace-pre-wrap">{note.content}</p>
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
