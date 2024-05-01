import { Link , useNavigate } from 'react-router-dom'

const Signup = () => {
  const navigate=useNavigate()
  function handleSubmit (evt) {
    evt.preventDefault()
    const users = JSON.parse(
      localStorage.getItem('USERS') || JSON.stringify([])
    )
    const user = Object.fromEntries(new FormData(evt.target).entries())
    localStorage.setItem('USERS', JSON.stringify([...users, user]))
    navigate('/')
  }
  return ( 
  <div className='flex flex-col items-center justify-center h-screen w-screen gap-25 max-w-6xl m-auto px-6'>
    <form  style={{ width: '100%', height: '100%', display:'flex', justifyContent:'center',alignItems:'center' , flexDirection:'column'}} onSubmit={handleSubmit}>
      <div>
        <h1 className='title-text-gradient text-5xl text-center' style={{ color: 'white' }}>CipherCanva</h1>
        <h3 className='font-extrabold text-xl opacity-80'>
          "Dive into the secrets of steganography with us – join our thriving
          online community!"
        </h3>
      </div>
      <div style={{ width:'100%'}} className='flex flex-col gap-4 w-96' >
        <div>
          <p className='font-bold text-xl opacity-80' style={{textAlign:'center',marginTop:'3%'}}>Signup to your Account</p>
          <div
            style={{
              display: 'flex',
              flexDirection: 'column',
              gap: 10,
              padding: 10,
              width: '100%',
              justifyContent: 'center',
              alignItems: 'center'
            }}
          >
            <input
            className='bg-transparent text-white border border-white p-4 rounded-md w-full'
              required
              style={{ padding: 10, width: '30%' }}
              type='text'
              placeholder='enter your name'
              name='name'
            />
            <input
            className='bg-transparent text-white border border-white p-4 rounded-md w-full'
              required
              style={{ padding: 10, width: '30%' }}
              type='email'
              placeholder='enter your email address'
              name='email'
            />
            <input
            className='bg-transparent text-white border border-white p-4 rounded-md w-full'
              required
              style={{ padding: 10, width: '30%' }}
              type='password'
              placeholder='enter your password'
              name='password'
            />
            <button  style={{ padding: 10, width: '30%' }} className='py-4 bg-white rounded-md text-black' >Signup</button>
          </div>
        </div>
        <div>
          <p style={linkStyle} >
          Already a user ? {' '} <Link className='font-bold' to='/'>Signin</Link>
        </p>
        </div>
      </div>
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

export default Signup
