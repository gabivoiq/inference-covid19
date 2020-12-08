let dropArea = document.getElementById('drop-area')

;['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
  dropArea.addEventListener(eventName, preventDefaults, false)
})

;['dragenter', 'dragover'].forEach(eventName => {
  dropArea.addEventListener(eventName, highlight, false)
})

;['dragleave', 'drop'].forEach(eventName => {
  dropArea.addEventListener(eventName, unhighlight, false)
})

function highlight(e) {
  dropArea.classList.add('highlight')
}

function unhighlight(e) {
  dropArea.classList.remove('highlight')
}

dropArea.addEventListener('drop', handleDrop, false)

function uploadFile(file) {
  let url = 'http://localhost:5000/uploader'
  let formData = new FormData()

  formData.append('file', file[0])

  fetch(url, {
    method: 'POST',
    body: formData
  })
  .then((res) => { console.log(res);

  res.json().then((res) => {
      console.log(res)
      let y = document.querySelector("#content")

      for (const [key, value] of Object.entries(res)) {
        console.log(`${key}: ${value}`);
        let x = document.createElement('div');
        x.innerHTML = `${key}: ${value}`;
        y.append(x);
      }

      // document.querySelector("#content").innerHTML = JSON.stringify(res);
  })})
  .catch(() => { /* Error. Inform the user */ })
}

function handleFiles(file) {
  uploadFile(file)
}

function handleDrop(e) {
  let dt = e.dataTransfer
  let files = dt.files

  handleFiles(files)
}

function preventDefaults (e) {
  e.preventDefault()
  e.stopPropagation()
}