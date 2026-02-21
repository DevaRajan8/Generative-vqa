
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from PIL import Image
import io
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
from ensemble_vqa_app import ProductionEnsembleVQA
from groq_service import get_groq_service
app = FastAPI(
    title="Ensemble VQA API",
    description="Visual Question Answering API with ensemble model routing",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
ensemble_model = None
groq_service = None
@app.on_event("startup")
async def startup_event():
    """Initialize the ensemble VQA model on server startup"""
    global ensemble_model, groq_service
    print("=" * 80)
    print("🚀 STARTING VQA API SERVER")
    print("=" * 80)
    BASE_CHECKPOINT = "./vqa_checkpoint.pt"
    SPATIAL_CHECKPOINT = "./vqa_spatial_checkpoint.pt"
    if not os.path.exists(BASE_CHECKPOINT):
        print(f"❌ Base checkpoint not found: {BASE_CHECKPOINT}")
        print("Please ensure vqa_checkpoint.pt is in the project root")
        sys.exit(1)
    if not os.path.exists(SPATIAL_CHECKPOINT):
        print(f"❌ Spatial checkpoint not found: {SPATIAL_CHECKPOINT}")
        print("Please ensure vqa_spatial_checkpoint.pt is in the project root")
        sys.exit(1)
    try:
        ensemble_model = ProductionEnsembleVQA(
            base_checkpoint=BASE_CHECKPOINT,
            spatial_checkpoint=SPATIAL_CHECKPOINT,
            device='cuda'
        )
        print("\n✅ VQA models loaded successfully!")
        try:
            groq_service = get_groq_service()
            print("✅ Groq LLM service initialized for accessibility features")
        except ValueError as e:
            print(f"⚠️  Groq service not available: {e}")
            print("   Accessibility descriptions will use fallback mode")
            groq_service = None
        print("📱 Mobile app can now connect")
        print("=" * 80)
    except Exception as e:
        print(f"\n❌ Failed to load models: {e}")
        sys.exit(1)
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Ensemble VQA API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "answer": "/api/answer (POST)"
        }
    }
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": ensemble_model is not None,
        "models": {
            "base": "loaded" if ensemble_model else "not loaded",
            "spatial": "loaded" if ensemble_model else "not loaded"
        }
    }
@app.post("/api/answer")
async def answer_question(
    image: UploadFile = File(...),
    question: str = Form(...)
):
    """
    Answer a visual question using the ensemble VQA system
    Args:
        image: Image file (JPEG, PNG)
        question: Question text
    Returns:
        JSON response with answer, model used, accessibility description, and metadata
    """
    if ensemble_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not question or question.strip() == "":
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    try:
        image_bytes = await image.read()
        try:
            pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")
        temp_image_path = "temp_upload.jpg"
        pil_image.save(temp_image_path)
        result = ensemble_model.answer(
            image_path=temp_image_path,
            question=question,
            use_beam_search=True,
            beam_width=5,
            verbose=True
        )
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        is_spatial = ensemble_model.is_spatial_question(question)
        description = None
        description_status = "not_available"
        if groq_service is not None:
            try:
                desc_result = groq_service.generate_description(
                    question=question,
                    answer=result['answer']
                )
                description = desc_result.get('description')
                description_status = desc_result.get('status', 'success')
            except Exception as e:
                print(f"⚠️  Groq description generation failed: {e}")
                description = f"Question: {question}. Answer: {result['answer']}."
                description_status = "fallback"
        else:
            description = f"Question: {question}. Answer: {result['answer']}."
            description_status = "fallback"
        reasoning_chain = None
        if result.get('kg_enhancement'):
            reasoning_chain = result.get('reasoning_chain', [])
        return JSONResponse(content={
            "success": True,
            "answer": result['answer'],
            "description": description,
            "description_status": description_status,
            "model_used": result['model_used'],
            "confidence": result['confidence'],
            "question_type": "spatial" if is_spatial else "general",
            "question": question,
            "kg_enhancement": result.get('kg_enhancement'),
            "reasoning_type": result.get('reasoning_type', 'neural'),
            "reasoning_chain": reasoning_chain,
            "metadata": {
                "beam_search": True,
                "beam_width": 5
            }
        })
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error processing request: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
@app.get("/api/models/info")
async def models_info():
    """Get information about loaded models"""
    if ensemble_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    return {
        "base_model": {
            "name": "Base VQA Model",
            "description": "General visual question answering",
            "accuracy": "50%",
            "use_case": "General questions about objects, colors, counts, etc."
        },
        "spatial_model": {
            "name": "Spatial Adapter Model",
            "description": "Spatial reasoning and positional questions",
            "accuracy": "40%",
            "use_case": "Spatial questions (left, right, above, below, etc.)"
        },
        "routing": {
            "method": "Keyword-based classification",
            "spatial_keywords": ensemble_model.SPATIAL_KEYWORDS
        },
        "conversation": {
            "enabled": ensemble_model.conversation_enabled if ensemble_model else False,
            "timeout_minutes": 30
        }
    }
@app.post("/api/conversation/answer")
async def answer_conversational(
    image: UploadFile = File(...),
    question: str = Form(...),
    session_id: str = Form(None)
):
    """
    Answer a visual question with multi-turn conversation support.
    Handles pronoun resolution and maintains conversation context.
    Args:
        image: Image file (JPEG, PNG)
        question: Question text (may contain pronouns like "it", "this")
        session_id: Optional session ID to continue conversation
    Returns:
        JSON response with answer, session_id, resolved question, and context
    """
    if ensemble_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not ensemble_model.conversation_enabled:
        raise HTTPException(
            status_code=501,
            detail="Conversational VQA not available. Use /api/answer instead."
        )
    if not question or question.strip() == "":
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    try:
        image_bytes = await image.read()
        try:
            pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")
        temp_image_path = "temp_upload.jpg"
        pil_image.save(temp_image_path)
        result = ensemble_model.answer_conversational(
            image_path=temp_image_path,
            question=question,
            session_id=session_id,
            use_beam_search=True,
            beam_width=5,
            verbose=True
        )
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        description = None
        if groq_service is not None:
            try:
                desc_result = groq_service.generate_description(
                    question=result['resolved_question'],
                    answer=result['answer']
                )
                description = desc_result.get('description')
            except:
                description = f"Question: {question}. Answer: {result['answer']}."
        else:
            description = f"Question: {question}. Answer: {result['answer']}."
        return JSONResponse(content={
            "success": True,
            "answer": result['answer'],
            "description": description,
            "session_id": result['session_id'],
            "resolved_question": result['resolved_question'],
            "original_question": question,
            "conversation_context": result['conversation_context'],
            "model_used": result['model_used'],
            "confidence": result['confidence'],
            "kg_enhancement": result.get('kg_enhancement'),
            "reasoning_type": result.get('reasoning_type', 'neural'),
            "reasoning_chain": result.get('reasoning_chain'),
            "metadata": {
                "beam_search": True,
                "beam_width": 5,
                "conversation_enabled": True
            }
        })
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error processing conversational request: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
@app.get("/api/conversation/{session_id}/history")
async def get_conversation_history(session_id: str):
    """
    Get conversation history for a session.
    Args:
        session_id: Session ID
    Returns:
        JSON with conversation history
    """
    if ensemble_model is None or not ensemble_model.conversation_enabled:
        raise HTTPException(status_code=503, detail="Conversation service not available")
    history = ensemble_model.conversation_manager.get_history(session_id)
    if history is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session {session_id} not found or expired"
        )
    return JSONResponse(content={
        "success": True,
        "session_id": session_id,
        "history": history,
        "turn_count": len(history)
    })
@app.delete("/api/conversation/{session_id}")
async def delete_conversation(session_id: str):
    """
    Delete a conversation session.
    Args:
        session_id: Session ID to delete
    Returns:
        JSON with success status
    """
    if ensemble_model is None or not ensemble_model.conversation_enabled:
        raise HTTPException(status_code=503, detail="Conversation service not available")
    deleted = ensemble_model.conversation_manager.delete_session(session_id)
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"Session {session_id} not found"
        )
    return JSONResponse(content={
        "success": True,
        "message": f"Session {session_id} deleted"
    })
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("🚀 ENSEMBLE VQA API SERVER")
    print("=" * 80)
    print("\n📋 Configuration:")
    print("  - Host: 0.0.0.0 (accessible from network)")
    print("  - Port: 8000")
    print("  - Reload: Enabled (development mode)")
    print("\n🔗 Access URLs:")
    print("  - Local: http://localhost:8000")
    print("  - Network: http://<your-ip>:8000")
    print("  - Docs: http://localhost:8000/docs")
    print("\n💡 For mobile testing:")
    print("  1. Find your local IP: ipconfig (Windows) or ifconfig (Mac/Linux)")
    print("  2. Update API_URL in mobile app to http://<your-ip>:8000")
    print("  3. Ensure phone and computer are on same network")
    print("=" * 80 + "\n")
    uvicorn.run(
        "backend_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )