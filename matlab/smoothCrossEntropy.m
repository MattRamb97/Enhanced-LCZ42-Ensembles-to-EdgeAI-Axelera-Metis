%SMOOTHCROSSENTROPY Label-smoothing + (optional) class weights (+ optional focal gamma)
%
%   Use with trainNetwork() as a drop-in replacement for classificationLayer.
%   When you train via a custom loop (dlnetwork), the layer will sit in the
%   graph but the loss is computed outside; that's fine (see tip below to
%   enable smoothing there too).
%
%   Constructor:
%       layer = smoothCrossEntropy(name, epsilon, classWeights, gamma)
%   where:
%       name          (char)   layer.Name
%       epsilon       (double) [0..0.2] label smoothing, default 0.05
%       classWeights  (1xK)    optional per-class weights
%       gamma         (double) focal gamma >=0 (0 disables), default 0
%
%   The layer expects inputs Y as probabilities (softmax before it).
%
% Matteo Rambaldi — Thesis utilities

classdef smoothCrossEntropy < nnet.layer.ClassificationLayer
    properties
        Epsilon (1,1) double {mustBeNonnegative, mustBeLessThanOrEqual(Epsilon,0.2)} = 0.05
        ClassWeights double = []   % 1 x K
        Gamma (1,1) double {mustBeNonnegative} = 0
    end

    methods
        function layer = smoothCrossEntropy(name, epsilon, classWeights, gamma)
            if nargin>=1 && ~isempty(name),        layer.Name = name;           end
            if nargin>=2 && ~isempty(epsilon),     layer.Epsilon = epsilon;     end
            if nargin>=3 && ~isempty(classWeights),layer.ClassWeights = classWeights(:)'; end
            if nargin>=4 && ~isempty(gamma),       layer.Gamma = gamma;         end
        end

        function loss = forwardLoss(layer, Y, T)
            % Accept Y as (1x1xKxN) or (KxN). Accept T as categorical or one-hot.
            % Convert to (K x N)
            Y = toKxN(Y);
            [K,N] = size(Y);

            Toh = toOneHot(T, K, N);  % K x N
            epss = layer.Epsilon;

            % label smoothing
            Tls = (1-epss)*Toh + (epss/K);

            % clamp probs
            Y = max(min(Y, 1 - 1e-7), 1e-7);

            % base cross-entropy (soft targets)
            nll = -sum(Tls .* log(Y), 1);  % 1 x N

            % optional focal modulation on hard target prob p_t
            if layer.Gamma > 0
                pt = sum(Toh .* Y, 1);  % prob assigned to the true class
                nll = (1 - pt).^layer.Gamma .* nll;
            end

            % optional class weights
            if ~isempty(layer.ClassWeights)
                cw = layer.ClassWeights(:)';                % 1 x K
                w  = cw * Toh;                              % 1 x N
                nll = w .* nll;
            end

            loss = mean(nll);
        end
    end
end

% ------------------------ helpers ------------------------
function Y = toKxN(Y)
    sz = size(Y);
    if numel(sz)==4
        % 1x1xKxN → KxN
        Y = reshape(Y, sz(3), sz(4));
    elseif numel(sz)==2
        % KxN already
    else
        % try squeeze and fall back
        Y = squeeze(Y);
        if ndims(Y)~=2
            error('Unexpected prediction size for Y.');
        end
    end
end

function Toh = toOneHot(T, K, N)
    if iscategorical(T)
        idx = grp2idx(T(:));
        Toh = zeros(K,N,'like',double(1));
        for i = 1:N, Toh(idx(i),i) = 1; end
    else
        sz = size(T);
        if numel(sz)==4
            % 1x1xKxN
            Toh = reshape(T, sz(3), sz(4));
        elseif numel(sz)==2
            % KxN
            Toh = T;
        else
            Toh = squeeze(T);
            if ndims(Toh)~=2
                error('Unexpected target size for T.');
            end
        end
        % ensure numeric
        if ~isfloat(Toh), Toh = single(Toh); end
    end
end