close all  % 关闭当前所有画的图
%% 速度
figure
plot(out.tout,out.Lv,'LineWidth',1);hold on
plot(out.tout,squeeze(out.f1v),'LineWidth',1);
plot(out.tout,squeeze(out.f2v),'LineWidth',1);
plot(out.tout,squeeze(out.f3v),'LineWidth',1);
legend('leader','1','2','3')
xlabel('t/s')
ylabel('速度 m/s')

%% 与前车间距
figure
plot(out.tout,out.Lv-squeeze(out.f1v),'LineWidth',1); hold on
plot(out.tout,squeeze(out.f1v)-squeeze(out.f2v),'LineWidth',1);
plot(out.tout,squeeze(out.f2v)-squeeze(out.f3v),'LineWidth',1);
legend('1','2','3')
xlabel('t/s')
ylabel('间距 m')


%% 加速度
a1=diff(squeeze(out.f1v))./diff(out.tout);
a2=diff(squeeze(out.f2v))./diff(out.tout);
a3=diff(squeeze(out.f3v))./diff(out.tout);
figure
plot(out.tout(1:end-1),a1,'LineWidth',1);hold on
plot(out.tout(1:end-1),a2,'LineWidth',1);hold on
plot(out.tout(1:end-1),a3,'LineWidth',1);hold on
legend('1','2','3')
xlabel('t/s')
ylabel('加速度 m/s^2')
