import React from 'react'
import { Link } from 'react-router-dom'
import { Heart, Activity } from 'lucide-react'
import ThemeToggle from './ThemeToggle' // Import ThemeToggle

const Header = () => {
  return (
    <header className="bg-white shadow-sm border-b border-gray-200 dark:bg-gray-800 dark:border-gray-700"> {/* Added dark mode styles */}
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          <Link to="/" className="flex items-center space-x-3">
            <div className="flex items-center space-x-2">
              <Heart className="h-8 w-8 text-red-500" />
              <Activity className="h-6 w-6 text-primary-500" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-gray-900 dark:text-white">ECG Classification AI</h1> {/* Added dark mode text color */}
              <p className="text-sm text-gray-600 dark:text-gray-300">AI-powered arrhythmia detection</p> {/* Added dark mode text color */}
            </div>
          </Link>
          
          <nav className="flex items-center space-x-6">
            <Link 
              to="/" 
              className="text-gray-700 hover:text-primary-600 font-medium transition-colors dark:text-gray-300 dark:hover:text-primary-500"> {/* Added dark mode text colors */}
              Upload ECG
            </Link>
            <Link 
              to="/admin" 
              className="text-gray-700 hover:text-primary-600 font-medium transition-colors dark:text-gray-300 dark:hover:text-primary-500">
              Admin Panel
            </Link>
            <Link 
              to="/view-ecg" 
              className="text-gray-700 hover:text-primary-600 font-medium transition-colors dark:text-gray-300 dark:hover:text-primary-500">
              View ECG
            </Link>
            <a 
              href="#about" 
              className="text-gray-700 hover:text-primary-600 font-medium transition-colors dark:text-gray-300 dark:hover:text-primary-500"> {/* Added dark mode text colors */}
              About
            </a>
            <a 
              href="#help" 
              className="text-gray-700 hover:text-primary-600 font-medium transition-colors dark:text-gray-300 dark:hover:text-primary-500"> {/* Added dark mode text colors */}
              Help
            </a>
            <ThemeToggle /> {/* Added ThemeToggle */}
          </nav>
        </div>
      </div>
    </header>
  )
}

export default Header