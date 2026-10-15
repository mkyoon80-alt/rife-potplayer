using System;
using System.Windows.Controls;
using System.Windows.Threading;

namespace RifeControl
{
    internal static class DiagnosticsText
    {
        public static void Update(TextBox box, string value, bool detachedPreview = false)
        {
            // The controller retains the latest report and retries on its next
            // timer tick after the tab is visible. Hidden tabs have no viewport.
            if (!detachedPreview && !box.IsVisible) return;
            if (box.Text == value) return;
            double vertical = box.VerticalOffset, horizontal = box.HorizontalOffset;
            int start = box.SelectionStart, length = box.SelectionLength;
            // Text replacement resets WPF's caret/viewport. Restore the user's
            // reading state without moving keyboard focus. Any final layout
            // restore below runs before the next dispatcher input turn.
            using (box.DeclareChangeBlock())
            {
                // Edit the existing text container. Assigning Text recreates
                // its view and can schedule a later caret-driven scroll.
                string previous = box.Text;
                int prefix = 0, suffix = 0;
                while (prefix < previous.Length && prefix < value.Length && previous[prefix] == value[prefix]) ++prefix;
                while (suffix < previous.Length - prefix && suffix < value.Length - prefix && previous[previous.Length - 1 - suffix] == value[value.Length - 1 - suffix]) ++suffix;
                box.Select(prefix, previous.Length - prefix - suffix);
                box.SelectedText = value.Substring(prefix, value.Length - prefix - suffix);
                start = Math.Min(start, value.Length);
                box.Select(start, Math.Min(length, value.Length - start));
            }
            // Finish text/caret layout before queuing the viewport restore.
            // Otherwise WPF can apply an older caret request after a new scroll.
            box.UpdateLayout();
            box.ScrollToHorizontalOffset(horizontal);
            box.ScrollToVerticalOffset(vertical);
            // WPF can finish restoring its text-view caret after a tab was
            // reattached. Loaded runs after layout, before the next Input turn.
            // A newer report or selection invalidates this one-shot restore.
            int savedStart = box.SelectionStart, savedLength = box.SelectionLength;
            box.Dispatcher.BeginInvoke(DispatcherPriority.Loaded, new Action(() => {
                if (box.IsVisible && box.Text == value && box.SelectionStart == savedStart && box.SelectionLength == savedLength)
                {
                    box.ScrollToHorizontalOffset(horizontal);
                    box.ScrollToVerticalOffset(vertical);
                }
            }));
        }
    }
}
