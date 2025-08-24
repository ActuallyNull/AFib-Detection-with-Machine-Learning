import React, { createContext, useContext, useReducer } from 'react'

const ECGContext = createContext()

const initialState = {
  ecgFiles: [],
  predictions: {},
  loading: false,
  error: null
}

const ecgReducer = (state, action) => {
  switch (action.type) {
    case 'ADD_ECG_FILES':
      return {
        ...state,
        ecgFiles: [...state.ecgFiles, ...action.payload]
      }
    case 'SET_PREDICTION':
      return {
        ...state,
        predictions: {
          ...state.predictions,
          [action.payload.id]: action.payload.prediction
        }
      }
    case 'SET_LOADING':
      return {
        ...state,
        loading: action.payload
      }
    case 'SET_ERROR':
      return {
        ...state,
        error: action.payload
      }
    case 'CLEAR_ERROR':
      return {
        ...state,
        error: null
      }
    default:
      return state
  }
}

export const ECGProvider = ({ children }) => {
  const [state, dispatch] = useReducer(ecgReducer, initialState)

  const addECGFiles = (files) => {
    dispatch({ type: 'ADD_ECG_FILES', payload: files })
  }

  const setPrediction = (id, prediction) => {
    dispatch({ type: 'SET_PREDICTION', payload: { id, prediction } })
  }

  const setLoading = (loading) => {
    dispatch({ type: 'SET_LOADING', payload: loading })
  }

  const setError = (error) => {
    dispatch({ type: 'SET_ERROR', payload: error })
  }

  const clearError = () => {
    dispatch({ type: 'CLEAR_ERROR' })
  }

  const value = {
    ...state,
    addECGFiles,
    setPrediction,
    setLoading,
    setError,
    clearError
  }

  return (
    <ECGContext.Provider value={value}>
      {children}
    </ECGContext.Provider>
  )
}

export const useECG = () => {
  const context = useContext(ECGContext)
  if (!context) {
    throw new Error('useECG must be used within an ECGProvider')
  }
  return context
}

