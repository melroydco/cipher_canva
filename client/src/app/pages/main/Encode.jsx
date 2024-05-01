import { useMutation } from '@tanstack/react-query'
import { clsx } from 'clsx'
import React, { useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'

export const Encode = () => {
  const navigate = useNavigate()
  const imageInputRef = useRef(null)
  const [text, setText] = useState('')
  const [file, setFile] = useState(null)
  const imageUrl = file ? URL.createObjectURL(file) : ''

  const maxChars = 300

  const handleChange = event => {
    const inputText = event.target.value
    if (inputText.length <= maxChars) {
      setText(inputText)
    }
  }

  const handleFileInputChange = event => {
    const file = event.target.files[0]
    setFile(file)
  }

  const mutation = useMutation({
    mutationFn: formData =>
      fetch('http://127.0.0.1:3030/image/encode', {
        method: 'POST',
        body: formData
      }).then(res => res.json()),
    onSuccess: data => {
      navigate(`/encode/success?downloadUrl=${data.url}`)
    }
  })

  function handleRemoveImage () {
    setFile(null)
    imageInputRef.current.value = null
  }

  function handleSubmit (evt) {
    evt.preventDefault()
    mutation.mutate(new FormData(evt.target))
  }
  return (
    <form
      className='flex flex-col max-w-7xl m-auto gap-12'
      onSubmit={handleSubmit}
    >
      <div className='flex flex-col gap-4'>
        <h1 className='title-text-gradient text-6xl text-center'>Encoder</h1>
        <p className='text-center'>
          Encode your secrets seamlessly by inserting your chosen image and
          corresponding plaintext on our steganography encoding page.
        </p>
      </div>

      <div className='flex justify-between items-center'>
        <label
          htmlFor='image'
          className={clsx(
            'cursor-pointer flex-col items-center',
            file ? 'hidden' : 'flex'
          )}
        >
          <img src='/uploadIcon.png' alt='upload' />
          <input
            name='image'
            id='image'
            type='file'
            onChange={handleFileInputChange}
            className='h-0 w-0'
            required
            ref={imageInputRef}
          />
          Upload image
        </label>

        <div
          className={clsx(file ? 'flex' : 'hidden')}
          style={{
            flexDirection: 'column',
            gap: 10,
            justifyContent: 'center',
            alignItems: 'center'
          }}
        >
          <img
            id='image'
            name='image'
            width={300}
            style={{
              objectFit: 'contain',
              marginTop: 10
            }}
            src={imageUrl}
            alt='selected'
          />
          <button
            type='button'
            disabled={mutation.isPending}
            onClick={handleRemoveImage}
          >
            Remove
          </button>
        </div>

        <div>
          <p>Enter plain text</p>
          <textarea
            disabled={mutation.isPending}
            name='text'
            style={{
              maxWidth: '100%',
              height: '100%',
              overflowY: 'auto',
              resize: 'none',
              padding: 10,
              color: '#222',
              background: '#FBFBFB',
              fontFamily: 'Poppins'
            }}
            required
            value={text}
            onChange={handleChange}
            maxLength={maxChars}
            rows={6} // Adjust the number of rows as needed
            cols={40} // Adjust the number of columns as needed
          />
          <p>
            Characters left: {maxChars - text.length}/{maxChars}
          </p>
        </div>
      </div>

      <button
        className='bg-[#222] p-4 rounded w-80 m-auto'
        type='submit'
        disabled={mutation.isPending}
      >
        {mutation.isPending ? 'Encoding...' : 'Encode'}
      </button>
    </form>
  )
}
