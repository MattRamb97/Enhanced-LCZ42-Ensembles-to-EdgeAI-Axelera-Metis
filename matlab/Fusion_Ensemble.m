% FUSION_ENSEMBLE  Fuse multiple teacher ensembles (MS/SAR) using probs.
%
% Usage (preferred, with calibration):
%   % Each model is a struct with:
%   %   .name        (string)         -- 'SAR', 'Rand', 'RandRGB', ...
%   %   .scoresCal   (Nc x K)         -- averaged softmax on CAL
%   %   .yCal        (Nc x 1 categorical)
%   %   .scoresAvg   (Nt x K)         -- averaged softmax on TEST
%   %   .yTrue       (Nt x 1 categorical)
%   %   .classes     (K-class names)
%   %
%   cfg.models = {resRand, resRandRGB, resSAR};
%   fusion = Fusion_Ensemble(cfg);
%
% Output:
%   fusion.sum, fusion.temp, fusion.stack  (each has .acc, .cm, .classes, .yTrue, .yPred)
%
% Matteo Rambaldi — Thesis utilities

function fusion = Fusion_Ensemble(cfg)

arguments
    cfg struct
end

models = cfg.models;
assert(iscell(models) && ~isempty(models), 'cfg.models must be a non-empty cell array.');

% ---- sanity: class alignment across models ----
K = size(models{1}.scoresAvg, 2);
classes = string(models{1}.classes);
for m = 2:numel(models)
    assert(size(models{m}.scoresAvg,2)==K, 'Class count mismatch across models.');
    assert(all(classes == string(models{m}.classes)), 'Class label mismatch across models.');
    % test labels must match and be aligned
    assert(isequal(models{1}.yTrue, models{m}.yTrue), ...
        'yTrue misaligned across models. Ensure same TEST order (same Index).');
end
yTrue = models{1}.yTrue;

% =========================================================
% 1) SUM-RULE (simple, robust)
% =========================================================
Psum = zeros(size(models{1}.scoresAvg),'single');
for m = 1:numel(models)
    Psum = Psum + normProbs(models{m}.scoresAvg);
end
Psum = Psum / numel(models);
sumRes = evaluate(Psum, yTrue, classes);
fprintf('[Fusion] SUM-RULE  Acc=%.4f\n', sumRes.acc);

% =========================================================
% 2) TEMPERATURE PER MODEL (requires CAL)
% =========================================================
haveCal = all(cellfun(@(s) isfield(s,'scoresCal') && isfield(s,'yCal'), models));
tempRes = struct([]);
if haveCal
    T = zeros(1,numel(models));
    for m = 1:numel(models)
        Pcal = normProbs(models{m}.scoresCal);
        T(m) = fitTemperature(Pcal, models{m}.yCal, classes);
        fprintf('[Fusion] %s temperature T=%.3f\n', nameOf(models{m}), T(m));
    end
    Ptemp = zeros(size(models{1}.scoresAvg),'single');
    for m = 1:numel(models)
        Pm = normProbs(models{m}.scoresAvg);
        Ptemp = Ptemp + applyTemperature(Pm, T(m));
    end
    Ptemp = Ptemp / numel(models);
    tempRes = evaluate(Ptemp, yTrue, classes);
    fprintf('[Fusion] TEMP      Acc=%.4f\n', tempRes.acc);
else
    fprintf('[Fusion] TEMP skipped (no scoresCal/yCal provided).\n');
end

% =========================================================
% 3) STACKING (multiclass logistic on CAL) (requires CAL)
% =========================================================
stackRes = struct([]);
if haveCal
    % Build stacking features by concatenating calibrated probs per model.
    % (Calibrate first for fair comparison; otherwise concat raw probs.)
    % We'll reuse the learned T from above if available, otherwise fit here.
    if isempty(tempRes)
        T = zeros(1,numel(models));
        for m = 1:numel(models)
            Pcal = normProbs(models{m}.scoresCal);
            T(m) = fitTemperature(Pcal, models{m}.yCal, classes);
        end
    end
    PcalAll = [];
    for m = 1:numel(models)
        Pm = normProbs(models{m}.scoresCal);
        PcalAll = [PcalAll, applyTemperature(Pm, T(m))]; %#ok<AGROW>
    end
    yCal = models{1}.yCal;  % must match across models by assertion above

    % Fit multinomial logistic regression (mnrfit expects numeric matrix + categorical response)
    B = mnrfit(double(PcalAll), categorical(yCal));

    % Apply on TEST
    PtestAll = [];
    for m = 1:numel(models)
        Pm = normProbs(models{m}.scoresAvg);
        PtestAll = [PtestAll, applyTemperature(Pm, T(m))]; %#ok<AGROW>
    end
    Pstack = mnrval(B, double(PtestAll));  % Nt x K
    stackRes = evaluate(single(Pstack), yTrue, classes);
    fprintf('[Fusion] STACKING  Acc=%.4f\n', stackRes.acc);
else
    fprintf('[Fusion] STACKING skipped (no scoresCal/yCal provided).\n');
end

% =========================================================
% 4) Pack results
% =========================================================
fusion = struct();
fusion.classes = classes;
fusion.sum   = sumRes;
fusion.temp  = tempRes;
fusion.stack = stackRes;

end

% ======================== HELPERS ========================

function nm = nameOf(s)
if isfield(s,'name'), nm = string(s.name); else, nm = "model"; end
end

function P = normProbs(Pin)
% Ensure numerical stability and row-normalization
Pin = single(Pin);
Pin = max(Pin, 1e-7);
Pin = Pin ./ sum(Pin,2);
P = Pin;
end

function res = evaluate(P, yTrue, classes)
[~, idx] = max(P,[],2);
yPred = categorical(idx, 1:numel(classes), classes);
acc = mean(yPred == yTrue);

% confusion matrix (row-normalized)
cm = confusionmat(yTrue, yPred, 'Order', categorical(classes));

res = struct();
res.acc     = acc;
res.cm      = cm;
res.yTrue   = yTrue;
res.yPred   = yPred;
res.classes = classes;
end

function T = fitTemperature(P, y, classes)
% Fit scalar temperature T to minimize NLL on calibration set
% We work in logit space by inverting softmax approximately with log.
% logits ~= log(P) up to a per-row constant; temperature rescales margins.
yIdx = double(grp2idx(categorical(y, classes)));
logP = log(max(P,1e-12));
% NLL(T) = -1/N sum_i log softmax(logP_i / T)[y_i]
nll = @(t) mean( -log( softmax_row(logP./t, yIdx) ) );
T0 = 1.0;
opts = optimset('Display','off');
T = fminbnd(nll, 0.2, 5.0, opts);
end

function p_y = softmax_row(S, yIdx)
% Return the softmax probability for the true class y_i per row
% S: NxK (logits), yIdx in [1..K]
S = double(S);
S = S - max(S,[],2);
expS = exp(S);
den = sum(expS,2);
lin = sub2ind(size(S), (1:size(S,1))', yIdx(:));
p_y = expS(lin) ./ den;
p_y = max(p_y, 1e-12);
end

function Pout = applyTemperature(Pin, T)
% Re-apply temperature by moving to (pseudo-)logits and back
S = log(max(Pin,1e-12));
S = S ./ T;
% softmax
S = S - max(S,[],2);
expS = exp(S);
Pout = expS ./ sum(expS,2);
Pout = single(Pout);
end