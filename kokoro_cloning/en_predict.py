import torch

def predict_stock(model, helper, common: dict[str, torch.Tensor], s_dec, s_pred, mapper, prompt):
    (text, lengths, mask) = (common['text'], common['lengths'], common['text_mask'])
    plbert = model.bert(text, attention_mask=(~mask).int())
    duration_embedding = model.bert_encoder(plbert).transpose(-1, -2)
    duration_embedding = mapper.shared.prosody(duration_embedding, mask, prompt)
    duration_context = model.predictor.text_encoder(duration_embedding, s_pred, lengths, mask)
    (duration_hidden, _) = model.predictor.lstm(duration_context)
    duration_logits = model.predictor.duration_proj(duration_hidden)
    duration_float = torch.sigmoid(duration_logits).sum(dim=-1).squeeze(0)
    duration = torch.round(duration_float).clamp(min=1).long()
    alignment = helper.make_alignment(duration)
    predicted_prosody = duration_context.transpose(-1, -2) @ alignment
    (predicted_f0, predicted_n) = model.predictor.F0Ntrain(predicted_prosody, s_pred)
    content = model.text_encoder(text, lengths, mask)
    content = mapper.shared.content(content, mask, prompt)
    full_wave = helper.safe_wave(model.decoder(content @ alignment, predicted_f0, predicted_n, s_dec).float()).squeeze()
    return {'duration_float': duration_float, 'duration': duration, 'predicted_f0': predicted_f0, 'predicted_n': predicted_n, 'full_wave': full_wave}
