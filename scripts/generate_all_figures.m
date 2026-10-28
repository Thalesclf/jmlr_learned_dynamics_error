% GENERATE_ALL_FIGURES  Figures 2 to 10 of "Realized Error Amplification in
% Learned Dynamical Systems", drawn in MATLAB (R2021a or newer).
%
% The plotted values are read from the tables in data/processed/figure_data/,
% written by scripts/export_matlab_plot_data.py, and the figures are saved as
% vector PDFs in figures/matlab/.
% Run the first section once. Every other section draws one figure and can
% then be run on its own (Ctrl+Enter); when sections are run on their own,
% the repository root must be the current folder. The first section changes
% graphics defaults of the MATLAB session; reset(groot) restores them.

%% Settings
root = fileparts(fileparts(mfilename('fullpath')));   % repository root (this file is in scripts/)
dataDir = fullfile(root, 'data', 'processed', 'figure_data');
outDir = fullfile(root, 'figures', 'matlab');
assert(isfolder(dataDir), 'No tables in %s: run python scripts/export_matlab_plot_data.py', dataDir);
if ~isfolder(outDir)
    mkdir(outDir);
end
set(groot, 'defaultTextInterpreter', 'latex', ...
    'defaultAxesTickLabelInterpreter', 'latex', ...
    'defaultLegendInterpreter', 'latex', ...
    'defaultAxesTitleHorizontalAlignment', 'left', ...   % panel letters on the left
    'defaultAxesXTickLabelRotation', 0, ...              % horizontal tick labels
    'defaultAxesBox', 'on', 'defaultAxesXGrid', 'on', 'defaultAxesYGrid', 'on', ...
    'defaultLineMarkerSize', 4);
% Legend icons 15 pt wide instead of 30 pt, so that the legends fit inside the panels
if isMATLABReleaseOlderThan('R2024b')
    set(groot, 'defaultLegendItemTokenSize', [15 18]);
else
    set(groot, 'defaultLegendIconColumnWidth', 15);
end

W = 6;                          % JMLR text width (in)
c = [0.165 0.471 0.839          % 1 blue
     0.922 0.408 0.204          % 2 orange
     0.106 0.686 0.478          % 3 green
     0.929 0.631 0              % 4 yellow
     0.910 0.482 0.643          % 5 pink
     0     0.514 0              % 6 dark green
     0.384 0.314 0.839          % 7 violet
     0.890 0.286 0.282          % 8 red
     0.322 0.318 0.306];        % 9 gray
mk = {'o', 's', '^', 'd', 'v', 'p', 'h', '*'};
fam = {'L96-20', 'L96-150', 'L96-2000', 'KS'};

%% Figure 2: error growth (fig_growth.pdf)
D = readtable(fullfile(dataDir, 'fig_growth.csv'));
figure('Units', 'inches', 'Position', [1 1 W 2.5]);
tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

% (a) Error relative to the saturation level: median and 10-90% band
nexttile;
hold on;
for k = 1:4     % bands first, so that no band covers a curve
    d = D(strcmp(D.family, fam{k}), :);
    fill([d.Lambda; flipud(d.Lambda)], [d.E_over_Esat_q10; flipud(d.E_over_Esat_q90)], ...
        c(k, :), 'FaceAlpha', 0.15, 'EdgeColor', 'none');
end
for k = 1:4
    d = D(strcmp(D.family, fam{k}), :);
    plot(d.Lambda, d.E_over_Esat_med, '-', 'Color', c(k, :), ...
        'Marker', mk{k}, 'MarkerFaceColor', c(k, :));
end
yline(1, 'Color', c(9, :));
set(gca, 'XScale', 'log', 'YScale', 'log', 'XLim', [1e-2 10]);
xlabel('$\Lambda$');
ylabel('$E_T/E_{\rm sat}$');
title('(a)');

% (b) First-order reconstruction relative to the error: median and 10-90% band
nexttile;
hold on;
for k = 1:4
    d = D(strcmp(D.family, fam{k}), :);
    fill([d.Lambda; flipud(d.Lambda)], [d.Rlin_q10; flipud(d.Rlin_q90)], ...
        c(k, :), 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');
end
for k = 1:4
    d = D(strcmp(D.family, fam{k}), :);
    plot(d.Lambda, d.Rlin_med, '-', 'Color', c(k, :), ...
        'Marker', mk{k}, 'MarkerFaceColor', c(k, :), 'DisplayName', fam{k});
end
yline(1, 'Color', c(9, :), 'HandleVisibility', 'off');
set(gca, 'XScale', 'log', 'YScale', 'log', 'XLim', [1e-2 10]);
xlabel('$\Lambda$');
ylabel('$R_{\rm lin}$');
title('(b)');
legend('Location', 'northwest');

exportgraphics(gcf, fullfile(outDir, 'fig_growth.pdf'), 'ContentType', 'vector');

%% Figure 3: scalar sensitivities (fig_scalar.pdf)
D = readtable(fullfile(dataDir, 'fig_scalar.csv'));
v = {'rho_Gmax', 'rho_forced_gain', 'rho_sigma1J', 'rho_S', 'rho_B', 'rho_Q', 'rho_L'};
names = {'$G_{\max}$', '$\|\mathcal{L}_T\|_2$', '$\sigma_1(\mathbf{J}^{\rm ref}_0)$', ...
         '$S_T$', '$B_T$', '$Q_T$', '$L_T$'};
styles = {'--', '--', '--', '-', '-', '-', '-'};  % dashed: learned dynamics only
cols = c([9 7 4 1 2 3 5], :);
panels = {'L96-2000', 'KS'};
titles = {'(a) L96-2000', '(b) KS'};
figure('Units', 'inches', 'Position', [1 1 W 2.7]);
tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
for p = 1:2
    % Median over seeds of the within-seed Spearman correlation with E_T
    nexttile;
    hold on;
    d = D(strcmp(D.family, panels{p}), :);
    h = gobjects(1, 7);
    for j = 1:7
        h(j) = plot(d.Lambda, d.(v{j}), styles{j}, 'Color', cols(j, :), ...
            'Marker', mk{j}, 'MarkerFaceColor', cols(j, :), 'DisplayName', names{j});
    end
    yline(0, 'Color', c(9, :));
    set(gca, 'XScale', 'log', 'XLim', [1e-2 10], 'YLim', [-0.6 1.05]);
    xlabel('$\Lambda$');
    title(titles{p});
    if p == 1
        ylabel('$\rho$');
        legend(h(1:3), 'Location', 'southwest');
    else
        set(gca, 'YTickLabel', []);
        legend(h(4:7), 'Location', 'southwest', 'NumColumns', 2);
    end
end

exportgraphics(gcf, fullfile(outDir, 'fig_scalar.pdf'), 'ContentType', 'vector');

%% Figure 4: error budget (fig_budget.pdf)
D = readtable(fullfile(dataDir, 'fig_budget.csv'));
f = {'S', 'gbar', 'eta', 'C', 'N'};
names = {'$S_T$', '$\bar{g}_T$', '$\eta_T$', '$C_T$', '$N_T$'};
panels = {'L96-2000', 'KS', 'L96-20'};
top = {'(a) L96-2000', '(b) KS', '(c) L96-20'};
bottom = {'(d)', '(e)', '(f)'};
figure('Units', 'inches', 'Position', [1 1 W 4.3]);
tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
for p = 1:3
    d = D(strcmp(D.family, panels{p}), :);

    % Top: geometric means over windows of the five factors and of E_T
    nexttile(p);
    hold on;
    for j = 1:5
        plot(d.Lambda, d.(f{j}), '-', 'Color', c(j, :), ...
            'Marker', mk{j}, 'MarkerFaceColor', c(j, :));
    end
    plot(d.Lambda, d.E, 'k-');
    set(gca, 'XScale', 'log', 'YScale', 'log', 'XLim', [1e-2 10], 'YLim', [1e-2 1e3], ...
        'XTickLabel', []);
    title(top{p});
    if p == 1
        ylabel('geometric mean');
    else
        set(gca, 'YTickLabel', []);
    end

    % Bottom: within-seed variance shares of log E_T (below half of the saturation level)
    nexttile(p + 3);
    hold on;
    for j = 1:5
        plot(d.Lambda, d.(['s_' f{j}]), '-', 'Color', c(j, :), ...
            'Marker', mk{j}, 'MarkerFaceColor', c(j, :), 'DisplayName', names{j});
    end
    yline(0, 'Color', c(9, :), 'HandleVisibility', 'off');
    set(gca, 'XScale', 'log', 'XLim', [1e-2 10], 'YLim', [-0.25 1.05]);
    xlabel('$\Lambda$');
    title(bottom{p});
    if p == 1
        ylabel('$s_j$');
    else
        set(gca, 'YTickLabel', []);
    end
end
plot(NaN, NaN, 'k-', 'DisplayName', '$E_T$');   % legend entry for the error of the top row
legend('Location', 'east');

exportgraphics(gcf, fullfile(outDir, 'fig_budget.pdf'), 'ContentType', 'vector');

%% Figure 5: directional efficiency (fig_efficiency.pdf)
D = readtable(fullfile(dataDir, 'fig_efficiency.csv'));
panels = {'L96-2000', 'KS'};
top = {'(a) L96-2000', '(b) KS'};
bottom = {'(c)', '(d)'};
figure('Units', 'inches', 'Position', [1 1 W 3.9]);
tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
for p = 1:2
    d = D(strcmp(D.family, panels{p}), :);

    % Top: median efficiency and the medians of the two null models
    nexttile(p);
    hold on;
    plot(d.Lambda, d.eta_med, '-o', 'Color', c(1, :), 'MarkerFaceColor', c(1, :), ...
        'DisplayName', '$\eta_T$');
    plot(d.Lambda, d.eta_Sigma_med, '--s', 'Color', c(2, :), 'MarkerFaceColor', c(2, :), ...
        'DisplayName', '$\eta^\Sigma_T$');
    plot(d.Lambda, d.eta_iso_med, ':^', 'Color', c(3, :), 'MarkerFaceColor', c(3, :), ...
        'DisplayName', '$\eta^{\rm iso}_T$');
    set(gca, 'XScale', 'log', 'XLim', [1e-2 10], 'YLim', [0 1.05], 'XTickLabel', []);
    title(top{p});
    if p == 1
        ylabel('$\eta_T$');
        legend('Location', 'northeast');
    else
        set(gca, 'YTickLabel', []);
    end

    % Bottom: orientation ratio, median and 10-90% band
    nexttile(p + 2);
    hold on;
    fill([d.Lambda; flipud(d.Lambda)], [d.alpha_q10; flipud(d.alpha_q90)], c(7, :), ...
        'FaceAlpha', 0.15, 'EdgeColor', 'none');
    plot(d.Lambda, d.alpha_med, '-o', 'Color', c(7, :), 'MarkerFaceColor', c(7, :));
    yline(1, 'Color', c(9, :));
    set(gca, 'XScale', 'log', 'XLim', [1e-2 10]);
    xlabel('$\Lambda$');
    title(bottom{p});
    if p == 1
        ylabel('$\alpha_T$');
    end
end

exportgraphics(gcf, fullfile(outDir, 'fig_efficiency.pdf'), 'ContentType', 'vector');

%% Figure 6: occupation of the singular spectrum (fig_spectrum.pdf)
D = readtable(fullfile(dataDir, 'fig_spectrum.csv'));
panels = {'L96-2000', 'KS'};
top = {'(a) L96-2000', '(b) KS'};
bottom = {'(c)', '(d)'};
figure('Units', 'inches', 'Position', [1 1 W 3.9]);
tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
for p = 1:2
    d = D(strcmp(D.family, panels{p}), :);

    % Top: number of directions carrying 90% of the transported energy
    nexttile(p);
    hold on;
    plot(d.Lambda, d.m90, '-o', 'Color', c(1, :), 'MarkerFaceColor', c(1, :), ...
        'DisplayName', 'actual defects');
    plot(d.Lambda, d.m90_Sigma, '--s', 'Color', c(2, :), 'MarkerFaceColor', c(2, :), ...
        'DisplayName', 'covariance null');
    plot(d.Lambda, d.m90_iso, ':^', 'Color', c(3, :), 'MarkerFaceColor', c(3, :), ...
        'DisplayName', 'isotropic null');
    set(gca, 'XScale', 'log', 'YScale', 'log', 'XLim', [1e-2 10], 'YLim', [1 20], ...
        'YTick', [1 2 5 10 20], 'YTickLabel', {'1', '2', '5', '10', '20'}, 'XTickLabel', []);
    title(top{p});
    if p == 1
        ylabel('$m_{90}$');
        legend('Location', 'southwest');
    else
        set(gca, 'YTickLabel', []);
    end

    % Bottom: participation number (no symbol for it in the text)
    nexttile(p + 2);
    hold on;
    plot(d.Lambda, d.Npart, '-o', 'Color', c(1, :), 'MarkerFaceColor', c(1, :));
    plot(d.Lambda, d.Npart_Sigma, '--s', 'Color', c(2, :), 'MarkerFaceColor', c(2, :));
    plot(d.Lambda, d.Npart_iso, ':^', 'Color', c(3, :), 'MarkerFaceColor', c(3, :));
    set(gca, 'XScale', 'log', 'YScale', 'log', 'XLim', [1e-2 10], 'YLim', [1 10], ...
        'YTick', [1 2 5 10], 'YTickLabel', {'1', '2', '5', '10'});
    xlabel('$\Lambda$');
    title(bottom{p});
    if p == 1
        ylabel('participation number');
    else
        set(gca, 'YTickLabel', []);
    end
end

exportgraphics(gcf, fullfile(outDir, 'fig_spectrum.pdf'), 'ContentType', 'vector');

%% Figure 7: coherence of the transported defects (fig_coherence.pdf)
D = readtable(fullfile(dataDir, 'fig_coherence.csv'));
panels = {'L96-2000', 'KS'};
titles = {'(a) L96-2000', '(b) KS'};
figure('Units', 'inches', 'Position', [1 1 W 2.5]);
tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
for p = 1:2
    d = D(strcmp(D.family, panels{p}), :);
    nexttile;
    hold on;
    fill([d.Lambda; flipud(d.Lambda)], [d.kappa_q10; flipud(d.kappa_q90)], c(1, :), ...
        'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    plot(d.Lambda, d.sqrtTeff_med, '--s', 'Color', c(9, :), 'MarkerFaceColor', c(9, :), ...
        'DisplayName', '$T_{\rm eff}^{1/2}$');
    plot(d.Lambda, d.kappa_med, '-o', 'Color', c(1, :), 'MarkerFaceColor', c(1, :), ...
        'DisplayName', '$\kappa_T$');
    plot(d.Lambda, d.kappa_iid_med, ':^', 'Color', c(2, :), 'MarkerFaceColor', c(2, :), ...
        'DisplayName', '$\kappa^{\rm iid}_T$');
    set(gca, 'XScale', 'log', 'YScale', 'log', 'XLim', [1e-2 10], 'YLim', [0.8 15], ...
        'YTick', [1 2 5 10], 'YTickLabel', {'1', '2', '5', '10'});
    xlabel('$\Lambda$');
    title(titles{p});
    if p == 1
        ylabel('$\kappa_T$');
        legend('Location', 'northwest');
    else
        set(gca, 'YTickLabel', []);
    end
end

exportgraphics(gcf, fullfile(outDir, 'fig_coherence.pdf'), 'ContentType', 'vector');

%% Figure 8: validity of the first-order reconstruction (fig_validity.pdf)
D = readtable(fullfile(dataDir, 'fig_validity.csv'));
figure('Units', 'inches', 'Position', [1 1 0.6 * W 2.6]);   % included at 0.6\textwidth
hold on;
for k = 1:4
    d = D(strcmp(D.family, fam{k}), :);
    plot(d.E_over_Esat, d.absLogRlin_med, '-', 'Color', c(k, :), ...
        'Marker', mk{k}, 'MarkerFaceColor', c(k, :), 'DisplayName', fam{k});
end
set(gca, 'XScale', 'log', 'YScale', 'log', 'XLim', [5e-4 20], 'YLim', [1e-6 1e2]);
xlabel('$E_T/E_{\rm sat}$');
ylabel('$|\log R_{\rm lin}|$');
legend('Location', 'northwest');

exportgraphics(gcf, fullfile(outDir, 'fig_validity.pdf'), 'ContentType', 'vector');

%% Figure 9: reference-free error estimates (fig_ensemble.pdf)
D = readtable(fullfile(dataDir, 'fig_ensemble.csv'));
v = {'rho_D', 'rho_Lhat', 'rho_Shat', 'rho_e0hat', 'rho_e0', 'rho_L', 'rho_Swhite', 'rho_Gmax_mdl'};
names = {'$D_T$', '$\hat{L}_T$', '$\hat{S}_T$', '$\|\hat{\mathbf{e}}_0\|_2$', ...
         '$\|\mathbf{e}_0\|_2$', '$L_T$', '$S^{\rm white}_T$', ...
         '$\sigma_1^2[\mathbf{\Phi}^{\rm mdl}(T,0)]$'};
styles = {'-o', '-s', '-^', '-d', '--v', '--', ':x', ':+'};
cols = c([1 5 3 4 9 6 8 7], :);
panels = {'L96-2000', 'L96-20', 'KS'};
top = {'(a) L96-2000', '(b) L96-20', '(c) KS'};
bottom = {'(d)', '(e)', '(f)'};
figure('Units', 'inches', 'Position', [1 1 W 4.8]);
tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
ax = gobjects(1, 3);
for p = 1:3
    d = D(strcmp(D.family, panels{p}), :);

    % Top: median over seeds of the within-seed Spearman correlation with E_T
    ax(p) = nexttile(p);
    hold on;
    for j = 1:8
        if all(isnan(d.(v{j})))     % the two statistical scores exist for L96-2000 only
            continue
        end
        plot(d.Lambda, d.(v{j}), styles{j}, 'Color', cols(j, :), ...
            'MarkerFaceColor', cols(j, :), 'DisplayName', names{j});
    end
    yline(0, 'Color', c(9, :), 'HandleVisibility', 'off');
    set(gca, 'XScale', 'log', 'XLim', [1e-2 10], 'YLim', [-0.4 1.05], 'XTickLabel', []);
    title(top{p});
    if p == 1
        ylabel('$\rho$');
    else
        set(gca, 'YTickLabel', []);
    end

    % Bottom: median score relative to the realized error (colors as in the top row)
    nexttile(p + 3);
    hold on;
    plot(d.Lambda, d.ratio_D, '-o', 'Color', cols(1, :), 'MarkerFaceColor', cols(1, :));
    plot(d.Lambda, d.ratio_Lhat, '-s', 'Color', cols(2, :), 'MarkerFaceColor', cols(2, :));
    yline(1, 'Color', c(9, :));
    set(gca, 'XScale', 'log', 'XLim', [1e-2 10], 'YLim', [0.5 2]);
    xlabel('$\Lambda$');
    title(bottom{p});
    if p == 1
        ylabel('score$/E_T$');
    else
        set(gca, 'YTickLabel', []);
    end
end
lgd = legend(ax(1));            % eight curves: the legend goes above the panels
lgd.NumColumns = 4;
lgd.Layout.Tile = 'north';

exportgraphics(gcf, fullfile(outDir, 'fig_ensemble.pdf'), 'ContentType', 'vector');

%% Figure 10: multistep fine-tuning (fig_intervention.pdf)
R = readtable(fullfile(dataDir, 'fig_intervention_change.csv'));
F = readtable(fullfile(dataDir, 'fig_intervention_factors.csv'));
A = readtable(fullfile(dataDir, 'fig_intervention_amplitude.csv'));
panels = {'L96-20', 'L96-2000', 'KS'};
row1 = {'(a) L96-20', '(b) L96-2000', '(c) KS'};
row2 = {'(d)', '(e)', '(f)'};
row3 = {'(g)', '(h)', '(i)'};
names = {'$S_T$', '$\bar{g}_T$', '$\eta_T$', '$C_T$', '$N_T$'};
arms = {'initial', 'one_step', 'rollout'};
armNames = {'initial', 'one-step', 'rollout'};
armStyles = {':o', '--s', '-^'};
armCols = c([9 1 2], :);
figure('Units', 'inches', 'Position', [1 1 W 6.6]);
tiledlayout(3, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
for p = 1:3
    % Row 1: paired change of the test relative RMSE per seed (bootstrap 95% intervals)
    nexttile(p);
    hold on;
    for s = 0:4
        d = R(strcmp(R.family, panels{p}) & R.seed == s, :);
        x = d.T * (1 + 0.04 * (s - 2));     % small shifts keep the seeds apart
        errorbar(x, d.change, d.change - d.ci_lo, d.ci_hi - d.change, mk{s + 1}, ...
            'Color', c(s + 1, :), 'MarkerFaceColor', c(s + 1, :), 'MarkerSize', 4, ...
            'CapSize', 0, 'DisplayName', sprintf('seed %d', s));
    end
    plot(d.T, d.change_median, 'k-', 'DisplayName', 'median');   % same in every seed's rows
    yline(0, 'Color', c(9, :), 'HandleVisibility', 'off');
    set(gca, 'XScale', 'log', 'XLim', [0.7 180], 'XTick', [1 4 16 64], ...
        'XTickLabel', {'1', '4', '16', '64'});
    title(row1{p});
    if p == 1
        ylabel('$\Delta$RMSE (\%)');
        lgd = legend;               % no room inside the panels: the legend goes above them
        lgd.NumColumns = 6;
        lgd.Layout.Tile = 'north';
    end

    % Row 2: median over seeds of the mean change of the log-factors and of log E_T
    nexttile(p + 3);
    hold on;
    d = F(strcmp(F.family, panels{p}), :);
    b = bar(1:7, [d.dS d.dgbar d.deta d.dC d.dN], 'EdgeColor', 'none');
    for j = 1:5
        b(j).FaceColor = c(j, :);
        b(j).DisplayName = names{j};
    end
    plot(1:7, d.dE, 'kd', 'MarkerFaceColor', 'k', 'DisplayName', '$E_T$');
    yline(0, 'Color', c(9, :), 'HandleVisibility', 'off');
    set(gca, 'XLim', [0.4 7.6], 'XTick', 1:2:7, 'XTickLabel', string(d.T(1:2:7)), 'XGrid', 'off');
    title(row2{p});
    if p == 1
        ylabel('$\Delta\log X_j$ (\%)');
    elseif p == 2
        legend('Location', 'southwest', 'NumColumns', 2);   % the panel with free space
    end

    % Row 3: median amplitude of the forecast relative to the reference state
    nexttile(p + 6);
    hold on;
    for j = 1:3
        d = A(strcmp(A.family, panels{p}) & strcmp(A.arm, arms{j}), :);
        plot(d.T, d.amp_ratio, armStyles{j}, 'Color', armCols(j, :), ...
            'MarkerFaceColor', armCols(j, :), 'DisplayName', armNames{j});
    end
    yline(1, 'Color', c(9, :), 'HandleVisibility', 'off');
    set(gca, 'XScale', 'log', 'XLim', [0.7 180], 'XTick', [1 4 16 64], ...
        'XTickLabel', {'1', '4', '16', '64'});
    xlabel('$T$');
    title(row3{p});
    if p == 1
        ylabel('amplitude ratio');
        legend('Location', 'northwest');
    end
end

exportgraphics(gcf, fullfile(outDir, 'fig_intervention.pdf'), 'ContentType', 'vector');
