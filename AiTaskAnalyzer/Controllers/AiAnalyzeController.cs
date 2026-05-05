using AiTaskAnalyzer.Interfaces;
using AiTaskAnalyzer.Models;
using AiTaskAnalyzer.Services;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;

namespace AiTaskAnalyzer.Controllers
{
    [ApiController]
    [Route("api/ai")]
    public class AiController : ControllerBase
    {
        private readonly LocalOnnxAiService _aiService;

        public AiController(LocalOnnxAiService aiService)
        {
            _aiService = aiService;
        }

        [HttpPost("analyze")]
        public ActionResult<AnalyzeResponse> Analyze([FromBody] AnalyzeRequest request)
        {
            if (string.IsNullOrWhiteSpace(request.Text))
                return BadRequest("Текст обращения клиента не может быть пустым.");

            var result = _aiService.AnalyzeClientRequest(request.Text);

            return Ok(new AnalyzeResponse
            {
                Result = result
            });
        }
    }
}
