using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Markup;
using System.Windows.Threading;

namespace RifeControl
{
    internal static class DiagnosticsScrollTests
    {
        public static object Run()
        {
            var rows = new List<object>();
            foreach (int width in new[] {464, 504})
            {
                Window window;
                using (var stream = Assembly.GetExecutingAssembly().GetManifestResourceStream("MainWindow.xaml"))
                    window = (Window)XamlReader.Load(stream);
                var controller = new Controller(window, true);
                controller.PreviewState("expanded");
                // A real WPF presentation source is needed: an unfocused,
                // detached text view does not execute deferred caret scrolling.
                window.WindowStartupLocation=WindowStartupLocation.Manual;
                window.Left=-20000;window.Top=-20000;window.ShowInTaskbar=false;window.ShowActivated=false;
                window.Show();
                var root = (FrameworkElement)window.Content;
                var box = (TextBox)window.FindName("DiagnosticsText");
                var tab = (TabItem)window.FindName("DiagnosticsTab");
                var copy = (Button)window.FindName("CopyButton");
                Action settle = () => {
                    for(int pass=0; pass<3; ++pass) {
                        root.Measure(new Size(width,501));root.Arrange(new Rect(0,0,width,501));root.UpdateLayout();
                        Dispatcher.CurrentDispatcher.Invoke(DispatcherPriority.ContextIdle,new Action(()=>{}));
                    }
                };
                Func<int,string> text = tick => {
                    var s = new StringBuilder("진단 갱신 시험 · 실제 재생 기록 아님\n");
                    for(int line=0; line<80; ++line)
                        s.AppendLine($"진단 {line:00} · 선택 유지 · 출력 프레임 {tick:000000} · TensorRT / HDR 상태 확인");
                    return s.ToString();
                };
                box.Text = text(0);settle();box.Focus();settle();
                if(box.ExtentHeight <= box.ViewportHeight)throw new Exception("Scroll fixture has no overflow.");
                box.Select(box.Text.IndexOf("진단 20",StringComparison.Ordinal),5);settle();
                box.ScrollToVerticalOffset(120);settle();double before=box.VerticalOffset;
                box.Text=text(1);settle();double legacyAfter=box.VerticalOffset;
                if(before<100 || legacyAfter>=before-1)throw new Exception($"Old text replacement did not reproduce the reported scroll reset: before={before} after={legacyAfter} focused={box.IsKeyboardFocused} extent={box.ExtentHeight} viewport={box.ViewportHeight}.");
                double maximumDrift=0;
                foreach(double requested in new[] {0.0,120.0,box.ExtentHeight-box.ViewportHeight})
                {
                    box.Select(box.Text.IndexOf("선택 유지",StringComparison.Ordinal),5);
                    if(requested==0)copy.Focus();else box.Focus();
                    box.ScrollToVerticalOffset(requested);settle();
                    double offset=box.VerticalOffset;int start=box.SelectionStart,length=box.SelectionLength;
                    var focus=FocusManager.GetFocusedElement(window);
                    for(int tick=2; tick<62; ++tick)
                    {
                        DiagnosticsText.Update(box,text(tick));settle();
                        maximumDrift=Math.Max(maximumDrift,Math.Abs(box.VerticalOffset-offset));
                        if(Math.Abs(box.VerticalOffset-offset)>0.5)throw new Exception("Live diagnostics moved the reading position.");
                        if(box.SelectionStart!=start || box.SelectionLength!=length || box.SelectedText!="선택 유지")throw new Exception("Live diagnostics replaced the user's selection.");
                        if(FocusManager.GetFocusedElement(window)!=focus)throw new Exception("Diagnostics changed the window's logical focus.");
                    }
                }
                // Real user input is a separate dispatcher operation after the
                // timer callback. Preserve WPF's render-before-input ordering.
                DiagnosticsText.Update(box,text(100));
                Dispatcher.CurrentDispatcher.BeginInvoke(DispatcherPriority.Input,new Action(()=>box.ScrollToVerticalOffset(60)));
                settle();
                if(Math.Abs(box.VerticalOffset-60)>0.5)throw new Exception($"Pending refresh overrode a later user scroll: offset={box.VerticalOffset}.");
                int events=0;TextChangedEventHandler changed=(s,e)=>++events;box.TextChanged+=changed;
                DiagnosticsText.Update(box,box.Text);settle();box.TextChanged-=changed;
                if(events!=0)throw new Exception("Unchanged diagnostics were replaced.");
                // Hidden tab refreshes must retain the reading state as well.
                double hiddenOffset=box.VerticalOffset;
                var setupTab=(TabItem)window.FindName("SetupTab");
                setupTab.Focus();setupTab.IsSelected=true;settle();
                DiagnosticsText.Update(box,text(101));settle();tab.Focus();tab.IsSelected=true;settle();
                if(Math.Abs(box.VerticalOffset-hiddenOffset)>0.5)throw new Exception("Tab switch lost the diagnostic reading position.");
                DiagnosticsText.Update(box,text(101));settle();
                if(box.Text!=text(101) || Math.Abs(box.VerticalOffset-hiddenOffset)>0.5)throw new Exception($"Visible refresh: updated={box.Text==text(101)}, before={hiddenOffset}, after={box.VerticalOffset}, visible={box.IsVisible}, loaded={box.IsLoaded}, tab={tab.IsSelected}.");
                box.Select(box.Text.Length-8,8);DiagnosticsText.Update(box,"중지됨");settle();
                if(box.SelectionStart>box.Text.Length || box.SelectionStart+box.SelectionLength>box.Text.Length || box.VerticalOffset!=0)throw new Exception("Short report left an invalid selection or scroll range.");
                DiagnosticsText.Update(box,"");settle();
                if(box.SelectionStart!=0 || box.SelectionLength!=0)throw new Exception("Empty report did not clamp the selection.");
                rows.Add(new {width,legacyBefore=before,legacyAfter,maximumDrift,refreshes=180,selectionPreserved=true,focusPreserved=true,newUserScrollWins=true,tabSwitch=true,shortAndEmpty=true});
                window.Close();
            }
            return new {passed=true,mode="actual WPF window/controls/dispatcher with text keyboard focus; offscreen, not physical mouse replay",cases=rows};
        }
    }
}
