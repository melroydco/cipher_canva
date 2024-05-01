import { AuthenticatedApp } from '../authenticated-app'
import { UnAuthenticatedApp } from '../unauthenticated-app'
import { useAuth } from './auth-context'

export const AppRouter = () => {
  const { user } = useAuth()
  if (user) return <AuthenticatedApp />
  return <UnAuthenticatedApp />
}
