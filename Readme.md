# Semantic Book Recommender System

## Project Overview
This project implements a sophisticated book recommendation system using Large Language Models (LLMs) and various Natural Language Processing (NLP) techniques. The system provides personalized book recommendations based on semantic search, sentiment analysis, and text classification.

## Components and Techniques

### 1. Data Exploration and Preprocessing (`data-exploration.ipynb`)
- Initial data analysis of the book dataset
- Data cleaning and preprocessing steps
- Feature extraction from book descriptions
- Handling missing values and data quality checks
- Visualization of data distributions and patterns

### 2. Vector Search Implementation (`vector-search.ipynb`)
#### Technique: Semantic Search with Embeddings
- Implementation of semantic search using embedding vectors
- Creation and management of a vector database using LangChain and Chroma
- Conversion of text descriptions into high-dimensional vectors
- Similarity search implementation using cosine similarity
- Enables natural language queries for book recommendations

**Theory**: Semantic search uses vector embeddings to capture the meaning of text in a high-dimensional space. Unlike traditional keyword search, this allows for finding similar books based on conceptual similarity rather than exact word matches. The system uses transformer-based models to generate these embeddings, enabling understanding of context and semantics.

### 3. Text Classification (`text-classification.ipynb`)
#### Technique: Zero-shot Classification
- Implementation of fiction/non-fiction classification
- Utilization of LLMs for zero-shot classification
- Genre categorization without traditional training data
- Integration with LangChain for streamlined classification

**Theory on Zero-Shot Classification**:
Zero-shot classification is an advanced machine learning paradigm that allows models to classify text into categories they haven't seen during training. Here's a deeper look at how it works:

1. **Traditional vs. Zero-shot Classification**:
   - Traditional classification requires extensive labeled training data for each category
   - Zero-shot can handle new categories without additional training
   - Utilizes the semantic understanding built into large language models

2. **Key Components**:
   - **Natural Language Inference (NLI)**: Models learn to understand relationships between text passages
   - **Cross-attention mechanisms**: Help models compare input text with category descriptions
   - **Semantic embeddings**: Convert both text and categories into comparable vector spaces

3. **Working Mechanism**:
   - The model receives:
     - Input text (book description)
     - Potential categories (fiction/non-fiction)
     - Category descriptions or examples
   - Creates embeddings for both input and categories
   - Computes compatibility scores between input and each category
   - Assigns the most probable category

4. **Advantages**:
   - Flexibility to handle new categories
   - No need for category-specific training data
   - Reduced computational resources
   - More natural handling of edge cases

5. **Applications in Book Classification**:
   - Genre classification
   - Theme identification
   - Content warning detection
   - Age group categorization

### 4. Sentiment Analysis (`sentiment-analysis.ipynb`)
#### Technique: Emotion Detection and Sentiment Analysis
- Advanced sentiment analysis of book descriptions
- Extraction of emotional tones (suspense, joy, sadness, etc.)
- Multi-label emotion classification
- Creation of emotional profiles for books

**Theory**: The sentiment analysis component uses transformer-based models to understand the emotional context and tone of book descriptions. This goes beyond basic positive/negative sentiment to capture complex emotional attributes, enabling recommendations based on emotional resonance.

### 5. Web Interface (`gradio-dashboard.py`)
- Interactive web application built with Gradio
- Integration of all components (search, classification, sentiment)
- User-friendly interface for book recommendations
- Real-time processing and recommendation generation

## Technical Requirements
- Python 3.11
- Key Dependencies:
  - langchain-community
  - langchain-opencv
  - langchain-chroma
  - transformers
  - gradio
  - pandas
  - matplotlib
  - seaborn

## Setup and Installation
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Create a `.env` file with your API keys (format below)
4. Download the required dataset from Kaggle
5. Run the notebooks in sequence to set up the components
6. Launch the Gradio dashboard using `python gradio-dashboard.py`

### Environment Configuration
Create a `.env` file in the root directory with the following format:
OPENAI_API_KEY = "Your OPENAI_API_KEY"
HUGGINGFACEHUB_API_TOKEN="YOUR HUGGINGFACE API KEY"

## How It Works
1. The system first processes book descriptions through the vector search component:
   - Text preprocessing and cleaning
   - Conversion to embeddings using transformer models
   - Storage in Chroma vector database for efficient retrieval

2. Books are classified into fiction/non-fiction categories:
   - Zero-shot classification using LLMs
   - Multiple category analysis for ambiguous cases
   - Confidence scoring for classification reliability

3. Sentiment analysis extracts emotional characteristics:
   - Multi-dimensional emotion analysis (joy, sadness, suspense, etc.)
   - Intensity scoring for each emotional aspect
   - Creation of emotional fingerprints for books

4. When a user makes a query:
   - The query is converted to a vector embedding using the same model
   - Similar books are found using cosine similarity measurements
   - Results are filtered based on classification and sentiment preferences
   - Top N recommendations are ranked by combined similarity scores
   - Recommendations are presented through the Gradio interface with relevant metadata

## Features
- Natural language book search with semantic understanding
- Fiction/non-fiction filtering with confidence scores
- Emotion-based recommendations with multiple dimensions:
  - Emotional intensity scoring
  - Mood-based filtering
  - Tone matching between query and books
- Similar book suggestions based on:
  - Plot similarities
  - Writing style matching
  - Thematic elements
- Interactive web interface with:
  - Real-time search results
  - Dynamic filtering options
  - Book cover displays
  - Description previews
- Real-time processing with:
  - Asynchronous query handling
  - Batch processing capabilities
  - Caching for frequent queries
  - Progressive loading of results

## Future Improvements
- Integration of more advanced LLM models:
  - Support for newer transformer architectures
  - Multi-modal models for cover image analysis
  - Fine-tuning on domain-specific book data

- Enhanced emotional analysis:
  - More granular emotion categories
  - Context-aware sentiment analysis
  - Writing style analysis
  - Character relationship mapping

- Additional book metadata integration:
  - Author writing style profiles
  - Series relationship mapping
  - Cross-reference with external book databases
  - Historical context integration

- Improved search algorithms:
  - Hybrid search combining semantic and keyword approaches
  - Personalized ranking based on user preferences
  - Dynamic re-ranking based on user interactions
  - Diversity-aware recommendation algorithms

- User experience enhancements:
  - Personalized user profiles
  - Reading history tracking
  - Social features for sharing recommendations
  - Custom collection creation

- Performance optimizations:
  - Distributed vector search
  - Improved caching strategies
  - Batch processing optimization
  - Response time improvements