%% generate_all_figures.m
% Regenerate all figures for the expanded JMLR manuscript from saved outputs.
%
% Expected project structure (edit paths below if needed):
%   results_l96_jmlr/seed_*/...
%   results_ks/seed_*/...
%   results_intervention_final_v3/{l96,ks}/seed_*/results.json (archived four-arm study)
%   results_rollout_retry_final/{l96,ks}/seed_*/results.json (included final Fig. 11)
%   results/seed_*/{perturbation_campaign.csv,epsilon_campaign.csv}   [legacy stress test]
%
% The script writes vector PDFs to figures/.
% Requires MATLAB Statistics and Machine Learning Toolbox for corr/prctile.

clear; close all; clc;

%% ---------------- USER SETTINGS -----------------------------------------
% Find input data relative to this source file rather than MATLAB's working folder.
rootDir = fileparts(mfilename('fullpath'));
l96Dir = fullfile(rootDir,"results_l96_jmlr");
ksDir  = fullfile(rootDir,"results_ks");
legacyDir = fullfile(rootDir,"results");
figDir = fullfile(rootDir,"figures");
if ~isfolder(figDir), mkdir(figDir); end

assert(isfolder(l96Dir),"Missing %s",l96Dir);
assert(isfolder(ksDir),"Missing %s",ksDir);
% The historical intervention dataset is archived; the retained Figure 11 instead
% uses results_rollout_retry_final. Do not substitute the historical dataset.

set(groot, ...
    'defaultAxesFontName','Helvetica', ...
    'defaultAxesTickLabelInterpreter','latex', ...
    'defaultLegendInterpreter','latex', ...
    'defaultTextInterpreter','latex');

%% ---------------- LOAD PRIMARY CAMPAIGNS --------------------------------
DL = readCampaign(l96Dir,"diagnostic_campaign.csv");
FL = readCampaign(l96Dir,"defect_terms_campaign.csv");
SL = readCampaign(l96Dir,"spectral_modes_campaign.csv");

DK = readCampaign(ksDir,"diagnostic_campaign.csv");
FK = readCampaign(ksDir,"defect_terms_campaign.csv");
SK = readCampaign(ksDir,"spectral_modes_campaign.csv");

fprintf("Loaded L96: D=%d F=%d S=%d\n",height(DL),height(FL),height(SL));
fprintf("Loaded KS : D=%d F=%d S=%d\n",height(DK),height(FK),height(SK));

%% FIG. 2 -- Local and worst-case finite-time diagnostics
f = paperFigure(520,220);
tl = tiledlayout(f,1,2,'TileSpacing','compact','Padding','compact');
plotLocalVsFinite(1,nexttile(tl),DL,"Lorenz--96 / Residual MLP");
plotLocalVsFinite(2,nexttile(tl),DK,"Kuramoto--Sivashinsky / FNO");
exportPaper(f,figDir,"fig2_local_vs_finite_time.pdf");

%% FIG. 3 -- Distribution of Gmax
f = paperFigure(520,220);
tl = tiledlayout(f,1,2,'TileSpacing','compact','Padding','compact');
plotGmaxBoxes(1,nexttile(tl),DL,"Lorenz--96 / residual MLP");
plotGmaxBoxes(2,nexttile(tl),DK,"Kuramoto--Sivashinsky / FNO");
exportPaper(f,figDir,"fig3_gmax_distribution.pdf");

%% FIG. 4 -- Error hierarchy
f = paperFigure(520,220);
tl = tiledlayout(f,1,2,'TileSpacing','compact','Padding','compact');
plotHierarchy(1,nexttile(tl),DL,"Lorenz--96 / residual MLP");
plotHierarchy(2,nexttile(tl),DK,"Kuramoto--Sivashinsky / FNO");
exportPaper(f,figDir,"fig4_error_hierarchy.pdf");

%% FIG. 5 -- Linear reconstruction and Rlin
f = paperFigure(520,410);
tl = tiledlayout(f,2,2,'TileSpacing','compact','Padding','compact');
plotReconScatter(1,nexttile(tl),DL,20,"Lorenz--96, $T=20$");
plotReconScatter(2,nexttile(tl),DK,16,"Kuramoto--Sivashinsky, $T=16$");
plotRlin(1,nexttile(tl),DL,"Lorenz--96");
plotRlin(2,nexttile(tl),DK,"Kuramoto--Sivashinsky");
exportPaper(f,figDir,"fig5_linear_reconstruction.pdf");

%% FIG. 6 -- Defect alignment versus gain efficiency
f = paperFigure(520,220);
tl = tiledlayout(f,1,2,'TileSpacing','compact','Padding','compact');
plotAlignmentDensity(1,nexttile(tl),FL,"Lorenz--96 / residual MLP");
plotAlignmentDensity(2,nexttile(tl),FK,"Kuramoto--Sivashinsky / FNO");
exportPaper(f,figDir,"fig6_alignment_efficiency.pdf");

%% FIG. 7 -- Error-aligned diagnostics
f = paperFigure(520,220);
tl = tiledlayout(f,1,2,'TileSpacing','compact','Padding','compact');
plotErrorAligned(1,nexttile(tl),DL,"Lorenz--96 / residual MLP");
plotErrorAligned(2,nexttile(tl),DK,"Kuramoto--Sivashinsky / FNO");
exportPaper(f,figDir,"fig7_error_aligned_diagnostics.pdf");

%% FIG. 8 -- Cancellation and spectral dimensionality
f = paperFigure(520,410);
tl = tiledlayout(f,2,2,'TileSpacing','compact','Padding','compact');
plotCancellation(1,nexttile(tl),DL,"Lorenz--96 / residual MLP");
plotCancellation(2,nexttile(tl),DK,"Kuramoto--Sivashinsky / FNO");
plotSpectralDim(1,nexttile(tl),FL,"Lorenz--96");
plotSpectralDim(2,nexttile(tl),FK,"Kuramoto--Sivashinsky");
exportPaper(f,figDir,"fig8_cancellation_spectral_dimensionality.pdf");

%% FIG. 9 -- Cumulative realized spectrum
f = paperFigure(520,220);
tl = tiledlayout(f,1,2,'TileSpacing','compact','Padding','compact');
plotCumulativeSpectrum(1,nexttile(tl),SL,[5 20 80 160],"Lorenz--96 / residual MLP");
plotCumulativeSpectrum(2,nexttile(tl),SK,[4 16 32 64],"Kuramoto--Sivashinsky / FNO");
exportPaper(f,figDir,"fig9_cumulative_spectrum.pdf");

%% FIG. 10 -- Model-trajectory finite-amplitude stress tests (legacy campaign)
if isfolder(legacyDir)
    try
        P = readCampaign(legacyDir,"perturbation_campaign.csv");
        E = readCampaign(legacyDir,"epsilon_campaign.csv");
        f = paperFigure(620,220);
        tl = tiledlayout(f,1,3,'TileSpacing','compact','Padding','compact');
        plotStressBoxes(nexttile(tl),P);
        plotStressAlignment(nexttile(tl),E);
        plotStressGain(nexttile(tl),E);
        exportPaper(f,figDir,"fig10_stress_tests.pdf");
    catch ME
        warning("Could not generate stress-test figure: %s",ME.message);
    end
else
    warning("Legacy results directory not found; skipping Fig. 10.");
end

%% FIG. 11 -- Final direct-rollout intervention (separate confirmatory campaign)
% Exact final five-seed experiment is now included under this directory.
rolloutRoot = fullfile(rootDir,"results_rollout_retry_final");
if isfolder(rolloutRoot) && ~isempty(dir(fullfile(rolloutRoot,"ks","seed_*","results.json"))) && ...
        ~isempty(dir(fullfile(rolloutRoot,"l96","seed_*","results.json")))
    generate_rollout_intervention_figures(rolloutRoot,figDir);
else
    warning("Fig. 11 NOT generated: missing results_rollout_retry_final/ks and/or l96 seed results. No substitute results used.");
end

fprintf("\nAll JMLR figures written to:\n  %s\n",figDir);

%% ======================== LOCAL FUNCTIONS ===============================
function T = readCampaign(resultsDir,fileName)
    listing = dir(fullfile(resultsDir,"seed_*",fileName));
    if isempty(listing), error("No %s under %s/seed_*",fileName,resultsDir); end
    tables = cell(numel(listing),1);
    for i=1:numel(listing)
        p=fullfile(listing(i).folder,listing(i).name);
        tables{i}=readtable(p,'VariableNamingRule','preserve');
        if ~ismember("seed",string(tables{i}.Properties.VariableNames))
            tok=regexp(listing(i).folder,'seed_(\d+)','tokens','once');
            if ~isempty(tok), tables{i}.seed=repmat(str2double(tok{1}),height(tables{i}),1); end
        end
    end
    T=vertcat(tables{:});
end

function name = errorColumn(D)
    vars=string(D.Properties.VariableNames);
    if ismember("rollout_error",vars), name="rollout_error";
    elseif ismember("actual_final_error",vars), name="actual_final_error";
    else, error("No rollout-error column found."); end
end

function r = spearman(x,y)
    q=isfinite(x)&isfinite(y); x=x(q); y=y(q);
    if numel(x)<2, r=NaN; else, r=corr(x,y,'Type','Spearman','Rows','complete'); end
end

function decorate(ax)
    grid(ax,'on'); box(ax,'on'); ax.FontSize=8.5; ax.LineWidth=.75; ax.GridAlpha=.15; ax.TickDir='out';
    set(ax,'TickLabelInterpreter','latex');
end

function plotLocalVsFinite(n,ax,D,titleText)
    hs=sort(unique(D.horizon)); err=D.(errorColumn(D));
    fields={"Gmax","spectral_radius_J","sigma1_J","fro_J"};
    labels={'$G_{\max}^{\rm ref}$','$\rho(J_t)$','$\sigma_1(J_t)$','$\|J_t\|_F$'};
    marks={'o','s','^','d'};
    hold(ax,'on');
    for j=1:numel(fields)
        vals=nan(size(hs));
        for i=1:numel(hs), q=D.horizon==hs(i); vals(i)=spearman(D.(fields{j})(q),err(q)); end
        h=semilogx(ax,hs,vals,['-' marks{j}],'DisplayName',labels{j}); h.MarkerFaceColor=h.Color;
    end
    set(gca,'XScale','log');
    yline(ax,0,'--','HandleVisibility','off'); ylim(ax,[-.2 .5]); xticks(ax,hs); xticklabels(ax,string(hs));
    xlabel(ax,'Forecast horizon ($T$)');

    if n == 1
        ylabel(ax,'Spearman correlation');
    elseif n == 2
        yticklabels('')
    end
    
    title(ax,titleText); decorate(ax);

    if contains(titleText,"Lorenz"), legend(ax,'Location','best','FontSize',7); end
end

function plotGmaxBoxes(n,ax,D,titleText)
    hs=sort(unique(D.horizon)); boxplot(ax,D.Gmax,D.horizon,'Positions',1:numel(hs),'Labels',string(hs),'Symbol','');
    set(ax,'YScale','log'); xlabel(ax,'Forecast horizon ($T$)');
    if n == 1
        ylabel(ax,'$G_{\max}^{\rm ref}$'); 
    else
        yticklabels('');
    end
    ylim([1 1e4]);
    title(ax,titleText); decorate(ax);
end

function plotHierarchy(n,ax,D,titleText)
    hs=sort(unique(D.horizon)); err=D.(errorColumn(D));
    fields={"Gmax","defect_l2_sum","propagated_defect_norm_sum","linear_predicted_error","defect_amplification_bound"};
    labels={'$G_{\max}^{\rm ref}$','$\sum_k\|e_k\|$','$\sum_k\|\Phi_ke_k\|$','$\|\sum_k\Phi_ke_k\|$','$B_T$'};
    hold(ax,'on');
    for j=1:numel(fields)
        vals=nan(size(hs)); for i=1:numel(hs),q=D.horizon==hs(i);vals(i)=spearman(D.(fields{j})(q),err(q));end
        h=semilogx(ax,hs,vals,'-o','DisplayName',labels{j}); h.MarkerFaceColor=h.Color;
    end
    set(gca,'XScale','log');
    ylim(ax,[0 1.03]); xticks(ax,hs); xticklabels(ax,string(hs)); xlabel(ax,'Forecast horizon ($T$)');
    if n == 1
        ylabel(ax,'Spearman correlation');
    elseif n == 2
        yticklabels({});
    end
    title(ax,titleText); decorate(ax);
    if contains(titleText,"Lorenz"), legend(ax,'Location','best','FontSize',7); end
end

function plotReconScatter(n,ax,D,T,titleText)
    q=D.horizon==T; x=D.(errorColumn(D))(q); y=D.linear_predicted_error(q);
    scatter(ax,x,y,8,'filled','MarkerFaceAlpha',.35); hold(ax,'on'); lo=min([x;y]); hi=max([x;y]); plot(ax,[lo hi],[lo hi],'--','LineWidth',1);
    set(ax,'XScale','linear','YScale','linear'); xlim(ax,'auto'); ylim(ax,'auto'); xlabel(ax,'Actual rollout error'); 
    if n == 1
        ylabel(ax,'$\|\delta_T^{\rm lin}\|_2$'); 
    end
    title(ax,titleText); decorate(ax);
end

function plotRlin(n,ax,D,titleText)
    hs=sort(unique(D.horizon)); med=nan(size(hs)); p10=med; p90=med;
    for i=1:numel(hs),v=D.linear_to_actual_ratio(D.horizon==hs(i));v=v(isfinite(v));med(i)=median(v);p10(i)=prctile(v,10);p90(i)=prctile(v,90);end
    hold(ax,'on'); fillBand(ax,hs,p10,p90,.16); h=semilogx(ax,hs,med,'-o'); h.MarkerFaceColor=h.Color; yline(ax,1,'--'); xticks(ax,hs); xticklabels(ax,string(hs)); xlabel(ax,'Forecast horizon ($T$)'); 
    if n == 1
        ylabel(ax,'$R_{\rm lin}$'); 
    end
    title(ax,titleText); decorate(ax);
    set(gca,'XScale','log');
end

function plotAlignmentDensity(n,ax,F,titleText)
    q=isfinite(F.tg_alignment)&isfinite(F.gain_efficiency); a=F.tg_alignment(q); eta=F.gain_efficiency(q);
    histogram2(ax,a,eta,80,'DisplayStyle','tile','ShowEmptyBins','off');
    colormap(jet); set(gca,'Layer','top');
    hold(ax,'on'); plot(ax,[0 1],[0 1],'--'); yline(ax,1,'--'); xlim(ax,[0 1]); ylim(ax,[0 1]); xlabel(ax,'Defect--TG alignment ($a_k$)'); 
    if n == 1
        ylabel(ax,'Gain efficiency ($\eta_k$)');
    else
        yticklabels('');
    end
    title(ax,titleText); c = colorbar(ax); c.TickLabelInterpreter='latex'; decorate(ax);
end

function plotErrorAligned(n,ax,D,titleText)
    hs=sort(unique(D.horizon)); err=D.(errorColumn(D));
    fields={"defect_l2_sum","propagated_defect_norm_sum","linear_predicted_error","effective_defect_gain"};
    labels={'$\sum_k\|e_k\|$','$\sum_k\|\Phi_ke_k\|$','$\|\sum_k\Phi_ke_k\|$','Effective defect gain'};
    if ismember("tg_alignment_weighted",string(D.Properties.VariableNames)), fields{end+1}="tg_alignment_weighted"; labels{end+1}='Weighted TG alignment'; end
    hold(ax,'on');
    for j=1:numel(fields)
        vals=nan(size(hs));for i=1:numel(hs),qq=D.horizon==hs(i);vals(i)=spearman(D.(fields{j})(qq),err(qq));end
        h=semilogx(ax,hs,vals,'-o','DisplayName',labels{j});h.MarkerFaceColor=h.Color;
    end
    ylim(ax,[-.15 1.03]);xticks(ax,hs);xticklabels(ax,string(hs));xlabel(ax,'Forecast horizon ($T$)');
    if n == 1
        ylabel(ax,'Spearman correlation');
    else
        yticklabels(ax,'');
    end
    set(ax,'XScale','log');
    title(ax,titleText);decorate(ax);
    if contains(titleText,"Lorenz"),legend(ax,'Location','best','FontSize',7);end
end

function plotCancellation(n,ax,D,titleText)
    hs=sort(unique(D.horizon));med=nan(size(hs));p10=med;p90=med;
    for i=1:numel(hs),v=D.defect_cancellation_ratio(D.horizon==hs(i));v=v(isfinite(v));med(i)=median(v);p10(i)=prctile(v,10);p90(i)=prctile(v,90);end
    hold(ax,'on');fillBand(ax,hs,p10,p90,.16);h=semilogx(ax,hs,med,'-o');h.MarkerFaceColor=h.Color;ylim(ax,[0 1.03]);xticks(ax,hs);xticklabels(ax,string(hs));
    set(gca,'XScale','log');
    if n == 1
        ylabel(ax,'Cancellation ratio ($C_T$)');
    else
        yticklabels(ax,'');
    end
    xticklabels(ax,'');
    title(ax,titleText);decorate(ax);
end

function plotSpectralDim(n,ax,F,titleText)
    q=isfinite(F.modes_for_90pct)&F.modes_for_90pct>0&isfinite(F.realized_participation_modes);F=F(q,:);hs=sort(unique(F.horizon));m90=nan(size(hs));np=nan(size(hs));
    for i=1:numel(hs),qq=F.horizon==hs(i);m90(i)=median(F.modes_for_90pct(qq));np(i)=median(F.realized_participation_modes(qq));end
    hold(ax,'on');h1=semilogx(ax,hs,m90,'-o','DisplayName','$m_{90}$');h1.MarkerFaceColor=h1.Color;h2=semilogx(ax,hs,np,'-s','DisplayName','$N_{\rm part}$');h2.MarkerFaceColor=h2.Color;xticks(ax,hs);xticklabels(ax,string(hs));xlabel(ax,'Forecast horizon ($T$)');
    set(gca,'XScale','log');
    if n == 1
        ylabel(ax,'Effective dimensionality');
    else
        yticklabels(ax,'');
    end
    legend(ax,'Location','best');decorate(ax);
end

function plotCumulativeSpectrum(n,ax,S,selectedT,titleText)
    hold(ax,'on');
    for j=1:numel(selectedT)
        T=selectedT(j); q=S.horizon==T&isfinite(S.cumulative_realized_fraction);SS=S(q,:);modes=sort(unique(SS.mode));med=nan(size(modes));
        for i=1:numel(modes),v=SS.cumulative_realized_fraction(SS.mode==modes(i));med(i)=median(v,'omitnan');end
        h=plot(ax,modes,med,'-o','MarkerSize',3,'DisplayName',sprintf('$T=%d$',T));h.MarkerFaceColor=h.Color;
    end
    yline(ax,.9,'--','HandleVisibility','off');ylim(ax,[0 1.02]);xlabel(ax,'Leading singular directions ($m$)');
    if n == 1
        ylabel(ax,'Cumulative realized energy ($P_m$)');
    else
        yticklabels(ax,'');
    end
    title(ax,titleText);legend(ax,'Location','southeast','FontSize',7);decorate(ax);
end

function plotStressBoxes(ax,P)
    Y=[P.G_TG,P.G_v2,P.G_random_mean,P.G_PGD];boxplot(ax,Y,'Labels',{'TG','$v_2$','Random','PGD'},'Symbol','');ylabel(ax,'Nonlinear gain ($G_{\rm NL}$)');title(ax,'$T=20$, $\epsilon=10^{-2}$');decorate(ax);
end

function plotStressAlignment(ax,E)
    epsv=sort(unique(E.epsilon));med=nan(size(epsv));p10=med;p90=med;
    for i=1:numel(epsv),v=1-E.alignment(E.epsilon==epsv(i));med(i)=median(v);p10(i)=prctile(v,10);p90(i)=prctile(v,90);end
    tiny=1e-16;med=max(med,tiny);p10=max(p10,tiny);p90=max(p90,tiny);hold(ax,'on');fillBand(ax,epsv,p10,p90,.16);h=loglog(ax,epsv,med,'-o');h.MarkerFaceColor=h.Color;xlabel(ax,'$\epsilon$');ylabel(ax,'$1-|v_{\rm TG}^Tv_{\rm adv}|$');decorate(ax);
end

function plotStressGain(ax,E)
    epsv=sort(unique(E.epsilon));hold(ax,'on');
    for pair={{"G_NL_TG","TG"},{"G_NL_PGD","PGD"}}
        key=pair{1}{1};lab=pair{1}{2};med=nan(size(epsv));for i=1:numel(epsv),q=E.epsilon==epsv(i);med(i)=median(E.(key)(q)./E.sigma1_sq(q));end
        h=semilogx(ax,epsv,med,'-o','DisplayName',lab);h.MarkerFaceColor=h.Color;
    end
    yline(ax,1,'--','DisplayName','Linear');xlabel(ax,'$\epsilon$');ylabel(ax,'$G_{\rm NL}/\sigma_1^2$');legend(ax,'Location','best');decorate(ax);
end

function fillBand(ax,x,lo,hi,alphaValue)
    q=isfinite(x)&isfinite(lo)&isfinite(hi);x=x(q);lo=lo(q);hi=hi(q);
    h=fill(ax,[x(:);flipud(x(:))],[lo(:);flipud(hi(:))],[.5 .5 .5],'EdgeColor','none','FaceAlpha',alphaValue,'HandleVisibility','off');uistack(h,'bottom');
end

function f=paperFigure(widthIn,heightIn)
    f=figure('Color','w','Units','points','Position',[1 1 widthIn heightIn],'PaperPositionMode','auto');
end

function exportPaper(f,figDir,fileName)
    drawnow;exportgraphics(f,fullfile(figDir,fileName),'ContentType','vector','BackgroundColor','white');close(f);
end

%% ====== FINAL-ROLLOUT INTERVENTION PLOTTING (merged source) ======
function generate_rollout_intervention_figures(rootDir, figDir)
%GENERATE_ROLLOUT_INTERVENTION_FIGURES
% Recreate the direct-rollout intervention figure used in the revised JMLR paper.
%
% Expected directory structure:
%   rootDir/
%       l96/seed_0/results.json
%       l96/seed_1/results.json
%       ...
%       ks/seed_0/results.json
%       ks/seed_1/results.json
%       ...
%
% Each results.json is expected to contain a JSON array of structs with fields:
%   system, seed, method, test
% where test contains keys such as h1_rel_rmse, h4_rel_rmse, ...
%
% This local function is called by the generate_all_figures script above.

if nargin < 1 || isempty(rootDir)
    rootDir = 'results_rollout_retry_final';
end
if nargin < 2 || isempty(figDir)
    figDir = 'figures';
end
if ~isfolder(rootDir)
    error('Root directory not found: %s', rootDir);
end
if ~isfolder(figDir)
    mkdir(figDir);
end

S = read_rollout_results(rootDir);
figPath = fullfile(figDir, 'fig11_rollout_intervention.pdf');
csvPath = fullfile(figDir, 'fig11_rollout_intervention_summary.csv');
texPath = fullfile(figDir, 'tab_intervention_rollout.tex');

summaryTbl = build_summary_table(S);
writetable(summaryTbl, csvPath);
write_latex_table(summaryTbl, texPath);

make_figure(S, figPath);

fprintf('Wrote figure to:\n  %s\n', figPath);
fprintf('Wrote CSV summary to:\n  %s\n', csvPath);
fprintf('Wrote LaTeX table snippet to:\n  %s\n', texPath);
end

function S = read_rollout_results(rootDir)
files = dir(fullfile(rootDir, '*', 'seed_*', 'results.json'));
if isempty(files)
    error('No results.json files found under %s/*/seed_*/', rootDir);
end

% Different result files may contain additional method-specific metadata.
% MATLAB cannot append structs with different field sets into one struct
% array, so retain only the four fields needed by this plotting script.
emptyItem = struct('system', '', 'seed', NaN, 'method', '', 'test', struct());
items = repmat(emptyItem, 0, 1);

for i = 1:numel(files)
    filePath = fullfile(files(i).folder, files(i).name);
    raw = fileread(filePath);
    decoded = jsondecode(raw);

    if isstruct(decoded)
        entries = num2cell(decoded(:));
    elseif iscell(decoded)
        entries = decoded(:);
    else
        error('Unexpected JSON structure in %s', filePath);
    end

    for j = 1:numel(entries)
        d = entries{j};
        required = {'system','seed','method','test'};
        for k = 1:numel(required)
            if ~isfield(d, required{k})
                error('Missing field "%s" in %s', required{k}, filePath);
            end
        end

        item = emptyItem;
        item.system = char(string(d.system));
        item.seed   = double(d.seed);
        item.method = char(string(d.method));
        item.test   = d.test;
        items(end+1,1) = item; %#ok<AGROW>
    end
end

S = items;
end

function tbl = build_summary_table(S)
horizons = extract_horizons(S(1).test);
systems = {'l96', 'ks'};
methods = {'one_step', 'rollout'};
rows = {};
for is = 1:numel(systems)
    sys = systems{is};
    for ih = 1:numel(horizons)
        H = horizons(ih);
        row = {upper(sys), H};
        for im = 1:numel(methods)
            med = median_metric(S, sys, methods{im}, H);
            row{end+1} = med; %#ok<AGROW>
        end
        pair = paired_change(S, sys, H);
        row{end+1} = pair; %#ok<AGROW>
        rows(end+1,:) = row; %#ok<AGROW>
    end
end

varNames = {'System','Horizon','OneStepRelRMSE','RolloutRelRMSE','PairedPercentChange'};
tbl = cell2table(rows, 'VariableNames', varNames);
end

function hs = extract_horizons(testStruct)
keys = fieldnames(testStruct);
hs = [];
for i = 1:numel(keys)
    tok = regexp(keys{i}, '^h(\d+)_rel_rmse$', 'tokens', 'once');
    if ~isempty(tok)
        hs(end+1) = str2double(tok{1}); %#ok<AGROW>
    end
end
hs = sort(hs);
end

function val = median_metric(S, systemName, methodName, H)
key = sprintf('h%d_rel_rmse', H);
vals = [];
for i = 1:numel(S)
    if strcmpi(S(i).system, systemName) && strcmpi(S(i).method, methodName) && isfield(S(i).test, key)
        vals(end+1) = S(i).test.(key); %#ok<AGROW>
    end
end
if isempty(vals)
    val = NaN;
else
    val = median(vals);
end
end

function pct = paired_change(S, systemName, H)
key = sprintf('h%d_rel_rmse', H);
seeds = unique([S(strcmpi({S.system}, systemName)).seed]);
vals = [];
for s = seeds
    oneIdx = find(strcmpi({S.system}, systemName) & strcmpi({S.method}, 'one_step') & [S.seed] == s, 1);
    rolIdx = find(strcmpi({S.system}, systemName) & strcmpi({S.method}, 'rollout') & [S.seed] == s, 1);
    if ~isempty(oneIdx) && ~isempty(rolIdx) && isfield(S(oneIdx).test, key) && isfield(S(rolIdx).test, key)
        base = S(oneIdx).test.(key);
        alt = S(rolIdx).test.(key);
        vals(end+1) = 100 * (alt / base - 1); %#ok<AGROW>
    end
end
if isempty(vals)
    pct = NaN;
else
    pct = median(vals);
end
end

function make_figure(S, figPath)
% Two-panel paper figure matching the manuscript caption.
% Left: marginal medians for both systems and both training arms.
% Right: median within-seed percentage change of rollout relative to one-step.
hs = [1 4 8 16 32 64];
fig = figure('Color', 'w', 'Units', 'points', 'Position', [100 100 620 245], 'PaperPositionMode', 'auto');
tl = tiledlayout(fig, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

ax = nexttile(tl);
plot_marginal_medians(ax, S, hs);

ax = nexttile(tl);
plot_paired_changes(ax, S, hs);

exportgraphics(fig, figPath, 'ContentType', 'vector', 'BackgroundColor', 'white');
close(fig);
end

function plot_marginal_medians(ax, S, hs)
l96One = arrayfun(@(h) median_metric(S, 'l96', 'one_step', h), hs);
l96Rol = arrayfun(@(h) median_metric(S, 'l96', 'rollout', h), hs);
ksOne  = arrayfun(@(h) median_metric(S, 'ks',  'one_step', h), hs);
ksRol  = arrayfun(@(h) median_metric(S, 'ks',  'rollout', h), hs);

h = loglog(ax, hs, l96One, '-o', 'LineWidth', 1.2, 'MarkerSize', 5, 'DisplayName', 'L96 one-step'); hold(ax, 'on');
h.MarkerFaceColor=h.Color;
h = loglog(ax, hs, l96Rol, '-s', 'LineWidth', 1.2, 'MarkerSize', 5, 'DisplayName', 'L96 rollout');
h.MarkerFaceColor=h.Color;
h = loglog(ax, hs, ksOne,  '-^', 'LineWidth', 1.2, 'MarkerSize', 5, 'DisplayName', 'KS one-step');
h.MarkerFaceColor=h.Color;
h = loglog(ax, hs, ksRol,  '-d', 'LineWidth', 1.2, 'MarkerSize', 5, 'DisplayName', 'KS rollout');
% h.MarkerFaceColor=h.Color;
xticks(ax, hs);
xticklabels(ax, string(hs));
set(gca,'YScale','linear');
xlabel(ax, 'Forecast horizon ($T$)', 'Interpreter', 'latex');
ylabel(ax, 'Median relative RMSE', 'Interpreter', 'latex');
legend(ax, 'Location', 'northwest', 'FontSize', 7);
grid(ax, 'on'); box(ax, 'on');
style_axes(ax);
end

function plot_paired_changes(ax, S, hs)
l96 = arrayfun(@(h) paired_change(S, 'l96', h), hs);
ks   = arrayfun(@(h) paired_change(S, 'ks', h), hs);

h = semilogx(ax, hs, l96, '-o', 'LineWidth', 1.2, 'MarkerSize', 5, 'DisplayName', 'L96'); hold(ax, 'on');
h.MarkerFaceColor=h.Color;
h = semilogx(ax, hs, ks,   '-s', 'LineWidth', 1.2, 'MarkerSize', 5, 'DisplayName', 'KS');
h.MarkerFaceColor=h.Color;
plot(ax, [hs(1) hs(end)], [0 0], 'k--', 'LineWidth', 0.8, 'HandleVisibility', 'off');
xticks(ax, hs);
xticklabels(ax, string(hs));
xlabel(ax, 'Forecast horizon ($T$)', 'Interpreter', 'latex');
ylabel(ax, 'Median paired change (\%)', 'Interpreter', 'latex');
legend(ax, 'Location', 'southwest', 'FontSize', 8);
grid(ax, 'on'); box(ax, 'on');
style_axes(ax);
end

function style_axes(ax)
ax.FontSize = 8.5;
ax.LineWidth = 0.75;
ax.TickDir = 'out';
ax.GridAlpha = 0.15;
end

function write_latex_table(tbl, texPath)
% Write a LaTeX snippet matching the paper table.
fid = fopen(texPath, 'w');
if fid < 0
    error('Could not open %s for writing.', texPath);
end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>

fprintf(fid, '%% Auto-generated by generate_rollout_intervention_figures.m\n');
fprintf(fid, '\\begin{tabular}{llrrrrrr}\n');
fprintf(fid, '\\toprule\n');
fprintf(fid, 'System & Quantity & $T=1$ & $T=4$ & $T=8$ & $T=16$ & $T=32$ & $T=64$ ');
fprintf(fid, '\\\\');
fprintf(fid, '\n');
fprintf(fid, '\\midrule\n');

systems = {'L96','KS'};
for is = 1:numel(systems)
    sysLabel = systems{is};
    q = strcmp(tbl.System, sysLabel);
    sub = tbl(q, :);
    one = sub.OneStepRelRMSE(:)';
    rol = sub.RolloutRelRMSE(:)';
    chg = sub.PairedPercentChange(:)';

    fprintf(fid, '%s & One-step rel. RMSE & %s ', sysLabel, join_num(one, false));
    fprintf(fid, '\\\\'); fprintf(fid, '\n');
    fprintf(fid, '   & Rollout rel. RMSE  & %s ', join_num(rol, false));
    fprintf(fid, '\\\\'); fprintf(fid, '\n');
    fprintf(fid, '   & Paired change (\\%%) & %s ', join_num(chg, true));
    fprintf(fid, '\\\\'); fprintf(fid, '\n');

    if is < numel(systems)
        fprintf(fid, '\\midrule\n');
    end
end

fprintf(fid, '\\bottomrule\n');
fprintf(fid, '\\end{tabular}\n');
end

function s = join_num(x, signed)
parts = strings(size(x));
for i = 1:numel(x)
    if signed
        parts(i) = sprintf('%+.3f', x(i));
    else
        if x(i) >= 0.1
            parts(i) = sprintf('%.5f', x(i));
        elseif x(i) >= 0.01
            parts(i) = sprintf('%.6f', x(i));
        else
            parts(i) = sprintf('%.6g', x(i));
        end
    end
end
s = strjoin(cellstr(parts), ' & ');
end
