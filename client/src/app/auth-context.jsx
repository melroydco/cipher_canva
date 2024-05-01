import React, { useState } from 'react'

const AuthContext = React.createContext(null)

export function AuthProvider ({ children }) {
  const [user, setUser] = useState(localStorage.getItem('USER'))

  const values = { user, setUser }

  return <AuthContext.Provider value={values}>{children}</AuthContext.Provider>
}

export function useAuth () {
  const context = React.useContext(AuthContext)
  if (!context) throw new Error('useAuth must be in scope with AuthProvider')
  return context
}
