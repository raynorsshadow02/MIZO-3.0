import pytest
import time
from unittest.mock import patch, AsyncMock
from app.services.llm_service import LLMService, LLMResult
from app.db.database import update_student, PROTOTYPE_STUDENT_ID


@pytest.mark.asyncio
async def test_route_1_hi():
    """Test 1: 'Hi' -> Casual simple route to local Ollama."""
    messages = [{"role": "user", "content": "Hi"}]
    with patch.object(LLMService, "_call_ollama", new_callable=AsyncMock) as mock_ollama, \
         patch.object(LLMService, "_call_groq", new_callable=AsyncMock) as mock_groq:
        mock_ollama.return_value = ("Hello! How can I help you today?", "llama3:latest")
        
        t0 = time.perf_counter()
        result = await LLMService.generate_response(messages=messages)
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        
        intent, complexity, primary, candidates = LLMService.classify_route(messages)
        assert primary == "ollama"
        assert result.provider == "ollama"
        assert result.is_fallback is False
        mock_ollama.assert_called_once()
        mock_groq.assert_not_called()
        print(f"\n[TEST 1] selected_route={primary} actual_provider={result.provider} external_api=False fallback={result.is_fallback} time={elapsed_ms}ms")


@pytest.mark.asyncio
async def test_route_2_thank_you():
    """Test 2: 'Thank you' -> Casual simple route to local Ollama."""
    messages = [{"role": "user", "content": "Thank you"}]
    with patch.object(LLMService, "_call_ollama", new_callable=AsyncMock) as mock_ollama, \
         patch.object(LLMService, "_call_groq", new_callable=AsyncMock) as mock_groq:
        mock_ollama.return_value = ("You're very welcome! Keep up the great work.", "llama3:latest")
        
        t0 = time.perf_counter()
        result = await LLMService.generate_response(messages=messages)
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        
        intent, complexity, primary, candidates = LLMService.classify_route(messages)
        assert primary == "ollama"
        assert result.provider == "ollama"
        assert result.is_fallback is False
        mock_ollama.assert_called_once()
        mock_groq.assert_not_called()
        print(f"\n[TEST 2] selected_route={primary} actual_provider={result.provider} external_api=False fallback={result.is_fallback} time={elapsed_ms}ms")


@pytest.mark.asyncio
async def test_route_3_what_is_my_name():
    """Test 3: 'What is my name?' -> Tier 0 Deterministic Database retrieval (Zero LLM)."""
    update_student(PROTOTYPE_STUDENT_ID, {"name": "Shahid"})
    messages = [{"role": "user", "content": "What is my name?"}]
    with patch.object(LLMService, "_call_ollama", new_callable=AsyncMock) as mock_ollama, \
         patch.object(LLMService, "_call_groq", new_callable=AsyncMock) as mock_groq, \
         patch.object(LLMService, "_call_qwen", new_callable=AsyncMock) as mock_qwen:
        
        t0 = time.perf_counter()
        result = await LLMService.generate_response(messages=messages)
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        
        assert result.provider == "system"
        assert "Shahid" in result.text
        assert result.is_fallback is False
        mock_ollama.assert_not_called()
        mock_groq.assert_not_called()
        mock_qwen.assert_not_called()
        print(f"\n[TEST 3] selected_route=tier0/deterministic actual_provider={result.provider} external_api=False fallback=None time={elapsed_ms}ms")


@pytest.mark.asyncio
async def test_route_4_what_is_your_name():
    """Test 4: 'What is your name?' -> Tier 0 Deterministic Mizo Identity (Zero LLM)."""
    messages = [{"role": "user", "content": "What is your name?"}]
    with patch.object(LLMService, "_call_ollama", new_callable=AsyncMock) as mock_ollama, \
         patch.object(LLMService, "_call_groq", new_callable=AsyncMock) as mock_groq, \
         patch.object(LLMService, "_call_qwen", new_callable=AsyncMock) as mock_qwen:
        
        t0 = time.perf_counter()
        result = await LLMService.generate_response(messages=messages)
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        
        assert result.provider == "system"
        assert "Mizo" in result.text
        assert result.is_fallback is False
        mock_ollama.assert_not_called()
        mock_groq.assert_not_called()
        mock_qwen.assert_not_called()
        print(f"\n[TEST 4] selected_route=tier0/deterministic actual_provider={result.provider} external_api=False fallback=None time={elapsed_ms}ms")


@pytest.mark.asyncio
async def test_route_5_simple_english_correction():
    """Test 5: 'Correct this sentence: I has a book.' -> Tier 1 Ollama."""
    messages = [{"role": "user", "content": "Correct this sentence: I has a book."}]
    with patch.object(LLMService, "_call_ollama", new_callable=AsyncMock) as mock_ollama, \
         patch.object(LLMService, "_call_groq", new_callable=AsyncMock) as mock_groq:
        mock_ollama.return_value = ("The correct sentence is: 'I have a book.'", "llama3:latest")
        
        t0 = time.perf_counter()
        result = await LLMService.generate_response(messages=messages)
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        
        intent, complexity, primary, candidates = LLMService.classify_route(messages)
        assert primary == "ollama"
        assert result.provider == "ollama"
        mock_ollama.assert_called_once()
        mock_groq.assert_not_called()
        print(f"\n[TEST 5] selected_route={primary} actual_provider={result.provider} external_api=False fallback={result.is_fallback} time={elapsed_ms}ms")


@pytest.mark.asyncio
async def test_route_6_simple_tutoring():
    """Test 6: 'What is gravity?' and 'Explain forward kinematics simply.' -> Tier 1 Ollama."""
    for query in ["What is gravity?", "Explain forward kinematics simply."]:
        messages = [{"role": "user", "content": query}]
        with patch.object(LLMService, "_call_ollama", new_callable=AsyncMock) as mock_ollama, \
             patch.object(LLMService, "_call_groq", new_callable=AsyncMock) as mock_groq:
            mock_ollama.return_value = (f"Simple explanation of {query}", "llama3:latest")
            
            t0 = time.perf_counter()
            result = await LLMService.generate_response(messages=messages)
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
            
            intent, complexity, primary, candidates = LLMService.classify_route(messages)
            assert primary == "ollama"
            assert result.provider == "ollama"
            mock_ollama.assert_called_once()
            mock_groq.assert_not_called()
            print(f"\n[TEST 6] query='{query}' selected_route={primary} actual_provider={result.provider} external_api=False fallback={result.is_fallback} time={elapsed_ms}ms")


@pytest.mark.asyncio
async def test_route_7_complex_technical_question():
    """Test 7: 'Derive the inverse kinematics equations for this specific 6-DOF manipulator and analyze singularities.' -> Tier 3 Groq."""
    query = "Derive the inverse kinematics equations for this specific 6-DOF manipulator and analyze singularities."
    messages = [{"role": "user", "content": query}]
    with patch.object(LLMService, "_call_groq", new_callable=AsyncMock) as mock_groq, \
         patch.object(LLMService, "_call_ollama", new_callable=AsyncMock) as mock_ollama:
        mock_groq.return_value = ("To derive the inverse kinematics for a 6-DOF manipulator...", "openai/gpt-oss-120b")
        
        t0 = time.perf_counter()
        result = await LLMService.generate_response(messages=messages)
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        
        intent, complexity, primary, candidates = LLMService.classify_route(messages)
        assert primary == "groq"
        assert result.provider == "groq"
        mock_groq.assert_called_once()
        mock_ollama.assert_not_called()
        print(f"\n[TEST 7] selected_route={primary} actual_provider={result.provider} external_api=True (Groq) fallback={result.is_fallback} time={elapsed_ms}ms")


@pytest.mark.asyncio
async def test_route_8_provider_failure_and_fallback():
    """Test 8: Failure and fallback behavior.
    - Normal simple task: Ollama fails -> cascades to OpenRouter free, NEVER Groq.
    - Complex task: Groq fails -> cascades to OpenRouter free / Ollama.
    """
    # 8A: Normal task fallback
    normal_messages = [{"role": "user", "content": "How do you form the past continuous tense?"}]
    with patch.object(LLMService, "_call_ollama", side_effect=Exception("Ollama local service unavailable")), \
         patch.object(LLMService, "_call_qwen", new_callable=AsyncMock) as mock_qwen, \
         patch.object(LLMService, "_call_groq", new_callable=AsyncMock) as mock_groq:
        mock_qwen.return_value = ("The past continuous tense is formed with was/were + verb-ing.", "openrouter/free")
        
        t0 = time.perf_counter()
        result_normal = await LLMService.generate_response(messages=normal_messages)
        elapsed_normal = round((time.perf_counter() - t0) * 1000, 2)
        
        assert result_normal.provider == "qwen"
        assert result_normal.is_fallback is True
        mock_groq.assert_not_called()
        print(f"\n[TEST 8A] normal_task: selected_route=ollama actual_provider={result_normal.provider} external_api=True (OpenRouter free) fallback=True (Ollama->OpenRouter) Groq_called=False time={elapsed_normal}ms")

    # 8B: Complex task fallback
    complex_messages = [{"role": "user", "content": "Derive the inverse kinematics equations for this specific 6-DOF manipulator and analyze singularities."}]
    with patch.object(LLMService, "_call_groq", side_effect=Exception("Groq 429 Rate Limit")), \
         patch.object(LLMService, "_call_qwen", new_callable=AsyncMock) as mock_qwen_c:
        mock_qwen_c.return_value = ("Analytical IK derivation for 6-DOF manipulator...", "openrouter/free")
        
        t0 = time.perf_counter()
        result_complex = await LLMService.generate_response(messages=complex_messages)
        elapsed_complex = round((time.perf_counter() - t0) * 1000, 2)
        
        assert result_complex.provider == "qwen"
        assert result_complex.is_fallback is True
        print(f"\n[TEST 8B] complex_task: selected_route=groq actual_provider={result_complex.provider} external_api=True (OpenRouter free fallback) fallback=True (Groq->OpenRouter) time={elapsed_complex}ms")
