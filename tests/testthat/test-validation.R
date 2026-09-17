# tests/testthat/test-validation.R
# ============================================================
# Unit tests for input validation (no API keys required)
# ============================================================

test_that("item.attributes must be a named list of character vectors", {

  # Should fail: unnamed list
  expect_error(
    AIGENIE:::items.attributes_validate(list(c("a", "b"))),
    regexp = "named"
  )

  # Should fail: single-element sublists
  expect_error(
    AIGENIE:::items.attributes_validate(list(trait = "only_one")),
    regexp = "two"
  )

})

test_that("embedding.model_validate recognises known providers", {

  expect_equal(
    AIGENIE:::embedding.model_validate("text-embedding-3-small"),
    "openai"
  )

  expect_equal(
    AIGENIE:::embedding.model_validate("jina-embeddings-v3"),
    "jina"
  )

  expect_equal(
    AIGENIE:::embedding.model_validate("BAAI/bge-small-en-v1.5"),
    "huggingface"
  )

})

test_that("detect_llm_provider resolves aliases correctly", {

  # Sonnet alias → full Anthropic model string
  result <- AIGENIE:::detect_llm_provider(
    "sonnet", anthropic.API = "fake-key-for-test"
  )
  expect_equal(result$provider, "anthropic")
  expect_match(result$model, "claude-sonnet")

  # gpt4o alias → gpt-4o
  result <- AIGENIE:::detect_llm_provider(
    "gpt4o", openai.API = "fake-key-for-test"
  )
  expect_equal(result$provider, "openai")
  expect_equal(result$model, "gpt-4o")

  # llama3 alias → groq
  result <- AIGENIE:::detect_llm_provider(
    "llama3", groq.API = "fake-key-for-test"
  )
  expect_equal(result$provider, "groq")

})

test_that("detect_llm_provider errors without the required API key", {

  expect_error(
    AIGENIE:::detect_llm_provider("gpt-4o"),
    regexp = "API key"
  )

  expect_error(
    AIGENIE:::detect_llm_provider("sonnet"),
    regexp = "API key"
  )

})

test_that("create_system.role returns a non-empty string", {

  sr <- AIGENIE:::create_system.role(
    domain = "psychology", scale.title = "Test",
    audience = NULL, response.options = NULL, system.role = NULL
  )
  expect_type(sr, "character")
  expect_true(nchar(sr) > 0)

})

test_that("create_main.prompts returns a named list matching item.attributes", {

  attrs <- list(trait_a = c("x", "y"), trait_b = c("a", "b"))
  prompts <- AIGENIE:::create_main.prompts(
    item.attributes = attrs,
    item.type.definitions = NULL,
    domain = "test", scale.title = "Test",
    prompt.notes = list(trait_a = "", trait_b = ""),
    audience = NULL, item.examples = NULL
  )

  expect_type(prompts, "list")
  expect_named(prompts, names(attrs))
  expect_true(all(nchar(unlist(prompts)) > 0))

})

test_that("EGA model validation preserves an explicitly requested model", {

  result <- AIGENIE:::validate_ega_params(
    EGA.algorithm = "walktrap",
    EGA.uni.method = "louvain",
    EGA_model = "TMFG"
  )

  expect_equal(result$EGA_model$type, "TMFG")
  expect_equal(result$EGA_model$overall, "TMFG")

  model_path <- tempfile(fileext = ".gguf")
  writeBin(as.raw(1), model_path)
  on.exit(unlink(model_path), add = TRUE)

  local_result <- suppressWarnings(
    AIGENIE:::validate_user_input_local_AIGENIE(
      item.attributes = list(
        conscientiousness = c(
          "organized", "responsible", "disciplined", "prudent"
        )
      ),
      model.path = model_path,
      embedding.model = "bert-base-uncased",
      main.prompts = NULL,
      temperature = 1,
      top.p = 1,
      target.N = 60,
      domain = "test",
      scale.title = "test",
      item.examples = NULL,
      audience = NULL,
      item.type.definitions = NULL,
      response.options = NULL,
      prompt.notes = NULL,
      system.role = NULL,
      EGA.model = "TMFG",
      EGA.algorithm = "walktrap",
      EGA.uni.method = "louvain",
      n.ctx = 2048,
      n.gpu.layers = -1,
      max.tokens = 384,
      device = "auto",
      batch.size = 32,
      pooling.strategy = "mean",
      max.length = 512L,
      keep.org = TRUE,
      items.only = FALSE,
      embeddings.only = FALSE,
      adaptive = TRUE,
      run.overall = FALSE,
      all.together = FALSE,
      plot = FALSE,
      silently = TRUE
    )
  )

  expect_equal(local_result$EGA.model$type, "TMFG")
  expect_equal(local_result$EGA.model$overall, "TMFG")

})

test_that("local generation exposes a reproducible llama.cpp seed", {

  expect_identical(formals(local_AIGENIE)$seed, 123L)
  expect_identical(
    formals(AIGENIE:::generate_items_via_local_llm)$seed,
    123L
  )
  expect_match(
    paste(
      deparse(body(AIGENIE:::generate_items_via_local_llm)),
      collapse = "\n"
    ),
    "seed = as.integer\\(seed\\)"
  )

})
