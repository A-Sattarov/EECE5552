% Example to get familiar with FallAllD dataset.
% By Majd SALEH 08-April-2020.

% load FallAllD if the structure doesn't exist in the workspace
if exist('FallAllD')~=1
    load('FallAllD.mat');
end

% Register keys
SubjectID=7;
ActivityID=102;
Device='Neck';
TrialNo=2;

Fs_Acc=238;% Sampling frequency of accelerometer 238 Hz
Fs_Gyr=238;% Sampling frequency of gyroscope 238 Hz
Fs_Mag=80;% Sampling frequency of magnetometer 80 Hz 
Fs_Bar=10;% Sampling frequency of barometer 10 Hz

Acc_Sen= 0.000244;% Accelerometer sensitivity 0.244 mg/LSB
Gyr_Sen= 0.07;    % Angular rate sensitivity 70 mdps/LSB
Mag_Sen= 0.00014; % Magnetic sensitivity 0.14 mgauss/LSB

CurReg=FallAllD(([FallAllD.SubjectID]==SubjectID)&([FallAllD.AtivityID]==ActivityID)&(strcmp({FallAllD.Device},Device))&([FallAllD.TrialNo]==TrialNo));
if isempty(CurReg)
    msgbox('No registers matches the selected keys!')
else
% Extract signals
Acc=CurReg.Acc;
Gyr=CurReg.Gyr;


% Data preprocessing
Acc=Acc*Acc_Sen;% convert to g units (m/s^2 units/9.81)
Gyr=Gyr*Gyr_Sen;% convert to dps units

t_Acc=(1:size(Acc,1))/Fs_Acc;
t_Gyr=(1:size(Gyr,1))/Fs_Gyr;


%% Plot
% plot acceleration
fig1=figure(1);
clf(fig1);
ax1(1) = subplot(4,1,1);  grid on; hold on;
plot(t_Acc, Acc(:,1),'Color',[217/256,83/256,25/256],'LineWidth',1);
plot(t_Acc, Acc(:,2),'Color',[0,1,0],'LineWidth',1);
plot(t_Acc, Acc(:,3),'Color',[0,114/256,189/256],'LineWidth',1);
ylabel('Acceleration (g)');
xticks([0:20]);
% Act=ActivityID2Str(ActivityID);
Act=int2str(ActivityID);
figTitle=['Subject ID=' num2str(SubjectID) ' , Device: ',Device,' ,' Act ' , Trial no= ' num2str(TrialNo)];
tit=title(figTitle);
tit.Interpreter='Latex';
tit.FontSize=14;
axis tight;
leg1 = legend(ax1(1), '$a_{x}$', '$a_{y}$', '$a_{z}$');
leg1.FontSize=14;
leg1.Interpreter='Latex';
xticklabels({})
% plot angular velocity
ax1(2) = subplot(4,1,2);  grid on; hold on;
plot(t_Gyr,Gyr(:,1),'Color',[217/256,83/256,25/256],'LineWidth',1);
plot(t_Gyr,Gyr(:,2),'Color',[0,1,0],'LineWidth',1);
plot(t_Gyr,Gyr(:,3),'Color',[0,114/256,189/256],'LineWidth',1);
ylabel('Angular velocity (dps)');
xticks([0:20]);
axis tight;
leg2 = legend(ax1(2), '$\omega_{x}$', '$\omega_{y}$', '$\omega_{z}$');
leg2.Interpreter='Latex';
leg2.FontSize=14;
xticklabels({})
% Link axes
linkaxes(ax1,'x');
end