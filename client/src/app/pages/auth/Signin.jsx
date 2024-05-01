import { Link } from 'react-router-dom'
import { useAuth } from '../../auth-context'

const Signin = () => {
  const { setUser } = useAuth()
  function handleSubmit (evt) {
    evt.preventDefault()
    const getUsers = JSON.parse(localStorage.getItem("USERS"))
    const User =  getUsers.find(user => user.email === evt.target[0].value)
    if(!User?.email){
      window.alert("Not a registered user")
      return
      }
    const formData = new FormData(evt.target)
    const user = Object.fromEntries(formData.entries())
    //localStorage.setItem('USER', JSON.stringify(user))
    setUser(user)
  }

  return (
    <div className='flex flex-col items-center justify-center h-screen w-screen gap-10 max-w-6xl m-auto px-6'>
      <div className='flex flex-col gap-6'>
        <h1 className='title-text-gradient text-5xl text-center'>
          CipherCanva
        </h1>
        <h2 className='font-extrabold text-xl opacity-80'>
          Welcome back to the CipherCanva Community
        </h2>
      </div>

      <form className='flex flex-col gap-4 w-96' onSubmit={handleSubmit}>
      <p className='font-bold text-xl opacity-80' style={{textAlign:'center',marginTop:'3%'}}>Signin to your Account</p>
        <input
          name='email'
          className='bg-transparent text-white border border-white p-4 rounded-md w-full'
          type='email'
          placeholder='Enter your email address'
          required
        />
        <input
          name='password'
          className='bg-transparent text-white border border-white p-4 rounded-md w-full'
          type='password'
          placeholder='Enter your password'
          required
        />
        <button type='submit' className='py-4 bg-white rounded-md text-black'>
          Login
        </button>
        <p style={linkStyle} >
          New user? <Link className='font-bold' to='/signup'>Signup</Link>
        </p>
      </form>
    </div>
  )
}
const linkStyle ={
  display:'flex',
  justifyContent: 'center',
  alignItems:'center',
  width:'100%'
  }

export default Signin
