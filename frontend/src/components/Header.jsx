import React from 'react'
import { Link } from 'react-router-dom'
import { Heart, Activity } from 'lucide-react'

const Header = () => {
  return (
    <header className="bg-white shadow-sm border-b border-gray-200">
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          <Link to="/" className="flex items-center space-x-3">
            <div className="flex items-center space-x-2">
              <Heart className="h-8 w-8 text-red-500" />
              <Activity className="h-6 w-6 text-primary-500" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-gray-900">ECG Classification AI</h1>
              <p className="text-sm text-gray-600">AI-powered arrhythmia detection</p>
            </div>
          </Link>
          
          <nav className="flex items-center space-x-6">
            <Link 
              to="/" 
              className="text-gray-700 hover:text-primary-600 font-medium transition-colors"
            >
              Upload ECG
            </Link>
            <a 
              href="#about" 
              className="text-gray-700 hover:text-primary-600 font-medium transition-colors"
            >
              About
            </a>
            <a 
              href="#help" 
              className="text-gray-700 hover:text-primary-600 font-medium transition-colors"
            >
              Help
            </a>
          </nav>
        </div>
      </div>
    </header>
  )
}

export default Header

