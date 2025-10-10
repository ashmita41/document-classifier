# Document Classifier Frontend

A modern, responsive web interface for testing the Document Classification API.

## Features

- **📤 Upload Documents** - Drag-and-drop or click to upload PDF files
- **📄 Document List** - View all uploaded documents with filtering and pagination
- **🔍 Search** - Full-text search across all documents
- **📊 Document Details** - View extracted sections, metadata, and Q&A pairs
- **💾 Export** - Export documents in multiple formats (JSON, CSV, Markdown, XML)
- **🔄 Reprocess** - Reprocess documents with latest models
- **🗑️ Delete** - Remove documents from the system

## Quick Start

### Option 1: Using Python's Built-in Server

```bash
cd frontend
python -m http.server 8080
```

Then open http://localhost:8080 in your browser.

### Option 2: Using Node.js http-server

```bash
# Install http-server globally (one time)
npm install -g http-server

# Run the server
cd frontend
http-server -p 8080 -c-1
```

### Option 3: Using the provided server script

```bash
cd frontend
python server.py
```

## Prerequisites

Make sure your backend API is running on http://localhost:8000

To start the backend:
```bash
cd ..
python -m uvicorn app.main:app --reload
```

Or use the startup scripts provided in the root directory.

## Usage

### Uploading Documents

1. Navigate to the **Upload** tab
2. Click the upload area or drag-and-drop a PDF file
3. Optionally select a document type
4. Click "Upload & Process"
5. Wait for processing to complete
6. Click "View Document" to see the results

### Viewing Documents

1. Navigate to the **Documents** tab
2. Use filters to narrow down results by status or type
3. Click on any document card to view details
4. Use pagination to browse through multiple pages

### Searching Documents

1. Navigate to the **Search** tab
2. Enter your search query
3. Press Enter or click the Search button
4. Click on any result to view document details

### Document Details

Once viewing a document, you can:
- See document information (status, type, size, pages, confidence)
- Browse through different tabs:
  - **Sections**: Hierarchical document structure
  - **Metadata**: Extracted dates, amounts, contacts, etc.
  - **Q&A Pairs**: Detected questions and answers
  - **Raw Data**: Complete JSON response
- Export the document in various formats
- Reprocess the document with latest models
- Delete the document

## API Configuration

The frontend is configured to connect to the backend at `http://localhost:8000`.

To change this, edit the `API_BASE_URL` constant in `app.js`:

```javascript
const API_BASE_URL = 'http://your-backend-url:port';
```

## Browser Support

- Chrome/Edge (recommended)
- Firefox
- Safari
- Opera

Modern browsers with ES6+ support required.

## Troubleshooting

### CORS Errors

If you see CORS errors in the console, make sure:
1. The backend is running
2. The backend has CORS enabled (it should by default)
3. You're accessing the frontend through a web server (not file://)

### Upload Fails

- Check file size (max 50MB)
- Ensure the file is a valid PDF
- Check backend logs for errors
- Verify MongoDB/database is running

### Documents Not Loading

- Verify the backend is running on port 8000
- Check browser console for errors
- Ensure you have documents in the database
- Try refreshing the page

## Development

### File Structure

```
frontend/
├── index.html      # Main HTML structure
├── styles.css      # Styling and responsive design
├── app.js          # Application logic and API calls
├── server.py       # Simple Python HTTP server
└── README.md       # This file
```

### Adding Features

To add new features:
1. Add UI elements to `index.html`
2. Add styling to `styles.css`
3. Add functionality to `app.js`
4. Follow existing patterns for API calls and error handling

### Styling

The frontend uses CSS custom properties (variables) for theming. To customize colors, edit the `:root` selector in `styles.css`.

## License

Same as the main project.

