using System;
using System.ClientModel.Primitives;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Azure;
using Azure.Core;
using Azure.ResourceManager;
using Azure.ResourceManager.CleanRoom;
using Azure.ResourceManager.CleanRoom.Models;
using Azure.ResourceManager.Resources;

return await CliApp.RunAsync(args);

internal static class CliApp
{
    public static async Task<int> RunAsync(string[] args)
    {
        var parsed = CliArgs.Parse(args);

        if (parsed.ShowHelp || parsed.Verb is null)
        {
            PrintHelp();
            return parsed.Verb is null && !parsed.ShowHelp ? 1 : 0;
        }

        try
        {
            var ctx = BuildContext(parsed);
            await DispatchAsync(ctx, parsed);
            return 0;
        }
        catch (RequestFailedException ex)
        {
            Console.Error.WriteLine($"Request failed (HTTP {ex.Status}): {ex.Message}");
            return ex.Status == 0 ? 1 : ex.Status;
        }
        catch (ArgumentException ex)
        {
            Console.Error.WriteLine($"Argument error: {ex.Message}");
            PrintHelp();
            return 2;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Error: {ex.Message}");
            return 1;
        }
    }

    private sealed record CmdContext(ArmClient Arm, string SubscriptionId);

    private static CmdContext BuildContext(CliArgs parsed)
    {
        TokenCredential credential = new Azure.Identity.DefaultAzureCredential();

        var subscriptionId =
            parsed.GetOption("--subscription")
            ?? Environment.GetEnvironmentVariable("AZURE_SUBSCRIPTION_ID")
            ?? throw new ArgumentException("--subscription <id> is required (or set AZURE_SUBSCRIPTION_ID).");

        var arm = new ArmClient(credential, subscriptionId);
        return new CmdContext(arm, subscriptionId);
    }

    private static async Task DispatchAsync(CmdContext ctx, CliArgs a)
    {
        switch ((a.Verb, a.SubVerb))
        {
            case ("collaborations", "create"):
                await CreateCollaborationAsync(ctx, a);
                break;
            case ("collaborations", "get"):
                await GetCollaborationAsync(ctx, a);
                break;
            case ("collaborations", "list"):
                await ListCollaborationsAsync(ctx, a);
                break;
            case ("collaborations", "delete"):
                await DeleteCollaborationAsync(ctx, a);
                break;
            case ("collaborations", "enable-workload"):
                await EnableWorkloadAsync(ctx, a);
                break;
            case ("collaborations", "add-collaborator"):
                await AddCollaboratorAsync(ctx, a);
                break;
            case ("collaborations", "recover"):
                await RecoverCollaborationAsync(ctx, a);
                break;
            case ("collaborations", "pause"):
                await PauseCollaborationAsync(ctx, a);
                break;
            case ("collaborations", "resume"):
                await ResumeCollaborationAsync(ctx, a);
                break;
            case ("collaborations", "get-readonly-kubeconfig"):
                await GetReadonlyKubeConfigAsync(ctx, a);
                break;

            case ("consortia", "create"):
                await CreateConsortiumAsync(ctx, a);
                break;
            case ("consortia", "get"):
                await GetConsortiumAsync(ctx, a);
                break;
            case ("consortia", "list"):
                await ListConsortiaAsync(ctx, a);
                break;
            case ("consortia", "delete"):
                await DeleteConsortiumAsync(ctx, a);
                break;

            default:
                throw new ArgumentException($"Unknown verb: '{a.Verb} {a.SubVerb}'.");
        }
    }

    private static async Task<ResourceGroupResource> GetResourceGroupAsync(CmdContext ctx, CliArgs a)
    {
        var rgName = a.GetOption("--resource-group")
            ?? Environment.GetEnvironmentVariable("AZURE_RESOURCE_GROUP")
            ?? throw new ArgumentException("--resource-group <name> is required (or set AZURE_RESOURCE_GROUP).");

        var sub = await ctx.Arm.GetDefaultSubscriptionAsync();
        return await sub.GetResourceGroupAsync(rgName);
    }

    // ---- Collaboration commands ----

    private static async Task CreateCollaborationAsync(CmdContext ctx, CliArgs a)
    {
        var name = a.Required(0, "collaborationName");
        var rg = await GetResourceGroupAsync(ctx, a);

        var location = new AzureLocation(
            a.GetOption("--location")
            ?? Environment.GetEnvironmentVariable("RP_LOCATION")
            ?? throw new ArgumentException("--location <region> is required."));

        var data = new CollaborationData(location);

        var resourceLocation = a.GetOption("--resource-location")
            ?? Environment.GetEnvironmentVariable("RESOURCE_LOCATION");
        if (!string.IsNullOrWhiteSpace(resourceLocation))
        {
            data.ResourceLocation = new AzureLocation(resourceLocation);
        }

        foreach (var email in a.GetOptionAll("--collaborator"))
        {
            data.Collaborators.Add(new Collaborator { UserIdentifier = email });
        }

        var op = await rg.GetCollaborations().CreateOrUpdateAsync(WaitUntil.Completed, name, data);
        WriteData(op.Value.Data);
    }

    private static async Task GetCollaborationAsync(CmdContext ctx, CliArgs a)
    {
        var name = a.Required(0, "collaborationName");
        var rg = await GetResourceGroupAsync(ctx, a);
        var collab = await rg.GetCollaborationAsync(name);
        WriteData(collab.Value.Data);
    }

    private static async Task ListCollaborationsAsync(CmdContext ctx, CliArgs a)
    {
        IAsyncEnumerable<CollaborationResource> source;
        if (a.GetOption("--resource-group") is not null
            || Environment.GetEnvironmentVariable("AZURE_RESOURCE_GROUP") is not null)
        {
            var rg = await GetResourceGroupAsync(ctx, a);
            source = rg.GetCollaborations().GetAllAsync();
        }
        else
        {
            var sub = await ctx.Arm.GetDefaultSubscriptionAsync();
            source = sub.GetCollaborationsAsync();
        }

        Console.WriteLine("[");
        bool first = true;
        await foreach (var item in source)
        {
            if (!first) Console.WriteLine(",");
            Console.Write(SerializeData(item.Data));
            first = false;
        }
        Console.WriteLine();
        Console.WriteLine("]");
    }

    private static async Task DeleteCollaborationAsync(CmdContext ctx, CliArgs a)
    {
        var name = a.Required(0, "collaborationName");
        var rg = await GetResourceGroupAsync(ctx, a);
        var collab = await rg.GetCollaborationAsync(name);
        await collab.Value.DeleteAsync(WaitUntil.Completed);
        Console.WriteLine($"Deleted collaboration '{name}'.");
    }

    private static async Task EnableWorkloadAsync(CmdContext ctx, CliArgs a)
    {
        var name = a.Required(0, "collaborationName");
        var rg = await GetResourceGroupAsync(ctx, a);
        var collab = await rg.GetCollaborationAsync(name);

        var workloadStr = a.GetOption("--workload-type") ?? nameof(WorkloadType.AnalyticsStrict);
        var workloadType = (WorkloadType)Enum.Parse(typeof(WorkloadType), workloadStr, ignoreCase: true);

        await collab.Value.EnableWorkloadAsync(
            WaitUntil.Completed,
            new EnableWorkloadContent(workloadType));

        // Re-fetch and print the latest collaboration state.
        var refreshed = await collab.Value.GetAsync();
        WriteData(refreshed.Value.Data);
    }

    private static async Task AddCollaboratorAsync(CmdContext ctx, CliArgs a)
    {
        var name = a.Required(0, "collaborationName");
        var rg = await GetResourceGroupAsync(ctx, a);
        var collab = await rg.GetCollaborationAsync(name);

        var user = a.GetOption("--user");
        var oid = a.GetOption("--object-id");
        var tid = a.GetOption("--tenant-id");

        if (string.IsNullOrWhiteSpace(user) && string.IsNullOrWhiteSpace(oid))
        {
            throw new ArgumentException("Either --user <email> or --object-id <oid> must be provided.");
        }

        var content = new AddCollaboratorContent
        {
            UserIdentifier = user,
            TenantId = tid,
            ObjectId = oid,
        };

        await collab.Value.AddCollaboratorAsync(WaitUntil.Completed, content);
        var refreshed = await collab.Value.GetAsync();
        WriteData(refreshed.Value.Data);
    }

    private static async Task RecoverCollaborationAsync(CmdContext ctx, CliArgs a)
    {
        var name = a.Required(0, "collaborationName");
        var rg = await GetResourceGroupAsync(ctx, a);
        var collab = await rg.GetCollaborationAsync(name);

        var force = a.GetBool("--force");
        await collab.Value.RecoverAsync(
            WaitUntil.Completed,
            new RecoverCollaborationContent(force));
        var refreshed = await collab.Value.GetAsync();
        WriteData(refreshed.Value.Data);
    }

    private static async Task PauseCollaborationAsync(CmdContext ctx, CliArgs a)
    {
        var name = a.Required(0, "collaborationName");
        var rg = await GetResourceGroupAsync(ctx, a);
        var collab = await rg.GetCollaborationAsync(name);
        await collab.Value.PauseAsync(WaitUntil.Completed);
        var refreshed = await collab.Value.GetAsync();
        WriteData(refreshed.Value.Data);
    }

    private static async Task ResumeCollaborationAsync(CmdContext ctx, CliArgs a)
    {
        var name = a.Required(0, "collaborationName");
        var rg = await GetResourceGroupAsync(ctx, a);
        var collab = await rg.GetCollaborationAsync(name);
        await collab.Value.ResumeAsync(WaitUntil.Completed);
        var refreshed = await collab.Value.GetAsync();
        WriteData(refreshed.Value.Data);
    }

    private static async Task GetReadonlyKubeConfigAsync(CmdContext ctx, CliArgs a)
    {
        var name = a.Required(0, "collaborationName");
        var rg = await GetResourceGroupAsync(ctx, a);
        var collab = await rg.GetCollaborationAsync(name);

        var result = await collab.Value.GetReadonlyKubeConfigAsync();
        var b64 = result.Value.Kubeconfig
            ?? throw new InvalidOperationException("Service returned an empty kubeconfig.");

        var bytes = Convert.FromBase64String(b64);
        var kubeconfig = Encoding.UTF8.GetString(bytes);

        var outPath = a.GetOption("--out");
        if (!string.IsNullOrWhiteSpace(outPath))
        {
            File.WriteAllText(outPath, kubeconfig, new UTF8Encoding(encoderShouldEmitUTF8Identifier: false));
            Console.WriteLine($"Wrote kubeconfig to {outPath}");
        }
        else
        {
            Console.Write(kubeconfig);
        }
    }

    // ---- Consortium commands ----

    private static async Task CreateConsortiumAsync(CmdContext ctx, CliArgs a)
    {
        var name = a.Required(0, "consortiumName");
        var rg = await GetResourceGroupAsync(ctx, a);

        var location = new AzureLocation(
            a.GetOption("--location")
            ?? Environment.GetEnvironmentVariable("RP_LOCATION")
            ?? throw new ArgumentException("--location <region> is required."));

        var data = new ConsortiumData(location);

        var resourceLocation = a.GetOption("--resource-location")
            ?? Environment.GetEnvironmentVariable("RESOURCE_LOCATION");
        if (!string.IsNullOrWhiteSpace(resourceLocation))
        {
            data.ResourceLocation = new AzureLocation(resourceLocation);
        }

        var op = await rg.GetConsortia().CreateOrUpdateAsync(WaitUntil.Completed, name, data);
        WriteData(op.Value.Data);
    }

    private static async Task GetConsortiumAsync(CmdContext ctx, CliArgs a)
    {
        var name = a.Required(0, "consortiumName");
        var rg = await GetResourceGroupAsync(ctx, a);
        var c = await rg.GetConsortiumAsync(name);
        WriteData(c.Value.Data);
    }

    private static async Task ListConsortiaAsync(CmdContext ctx, CliArgs a)
    {
        IAsyncEnumerable<ConsortiumResource> source;
        if (a.GetOption("--resource-group") is not null
            || Environment.GetEnvironmentVariable("AZURE_RESOURCE_GROUP") is not null)
        {
            var rg = await GetResourceGroupAsync(ctx, a);
            source = rg.GetConsortia().GetAllAsync();
        }
        else
        {
            var sub = await ctx.Arm.GetDefaultSubscriptionAsync();
            source = sub.GetConsortiaAsync();
        }

        Console.WriteLine("[");
        bool first = true;
        await foreach (var item in source)
        {
            if (!first) Console.WriteLine(",");
            Console.Write(SerializeData(item.Data));
            first = false;
        }
        Console.WriteLine();
        Console.WriteLine("]");
    }

    private static async Task DeleteConsortiumAsync(CmdContext ctx, CliArgs a)
    {
        var name = a.Required(0, "consortiumName");
        var rg = await GetResourceGroupAsync(ctx, a);
        var c = await rg.GetConsortiumAsync(name);
        await c.Value.DeleteAsync(WaitUntil.Completed);
        Console.WriteLine($"Deleted consortium '{name}'.");
    }

    // ---- Output helpers ----

    private static void WriteData<T>(T model) where T : IPersistableModel<T>
    {
        Console.WriteLine(SerializeData(model));
    }

    private static string SerializeData<T>(T model) where T : IPersistableModel<T>
    {
        var raw = ModelReaderWriter.Write(model).ToString();
        try
        {
            using var doc = JsonDocument.Parse(raw);
            return JsonSerializer.Serialize(doc.RootElement, new JsonSerializerOptions { WriteIndented = true });
        }
        catch
        {
            return raw;
        }
    }

    private static void PrintHelp()
    {
        var help = """
            Azure Cleanroom Management Plane - CLI

              Usage:
                cleanroom-mgmt [global-options] <resource> <action> [args...] [options]

              Authentication:
                Uses Azure.Identity DefaultAzureCredential — sign in with `az login`
                (or set AZURE_CLIENT_ID / AZURE_TENANT_ID / AZURE_CLIENT_SECRET for SPN,
                or run under a managed identity).

              Global options:
                --subscription <id>      Azure subscription id (env: AZURE_SUBSCRIPTION_ID)
                --resource-group <name>  Resource group (env: AZURE_RESOURCE_GROUP)
                -h | --help              Show this help

              Collaboration verbs:
                collaborations create   <name>  --location <region> [--resource-location <region>]
                                                [--collaborator <email>]...
                collaborations get      <name>
                collaborations list                                       (defaults to subscription if no --resource-group)
                collaborations delete   <name>
                collaborations enable-workload <name> [--workload-type AnalyticsStrict]
                collaborations add-collaborator <name>  --user <email> | --object-id <oid> [--tenant-id <tid>]
                collaborations recover  <name> [--force]
                collaborations pause    <name>
                collaborations resume   <name>
                collaborations get-readonly-kubeconfig <name> [--out <path>]

              Consortium verbs:
                consortia      create   <name>  --location <region> [--resource-location <region>]
                consortia      get      <name>
                consortia      list                                       (defaults to subscription if no --resource-group)
                consortia      delete   <name>

              Environment:
                AZURE_SUBSCRIPTION_ID    Default subscription id
                AZURE_RESOURCE_GROUP     Default resource group
                RP_LOCATION              Default ARM RP location for create
                RESOURCE_LOCATION        Default region for actual resource deployment
            """;
        Console.WriteLine(help);
    }
}

internal sealed class CliArgs
{
    public string? Verb { get; private set; }
    public string? SubVerb { get; private set; }
    public bool ShowHelp { get; private set; }

    private readonly List<string> _positional = new();
    private readonly Dictionary<string, List<string?>> _options = new(StringComparer.OrdinalIgnoreCase);

    public static CliArgs Parse(string[] args)
    {
        var r = new CliArgs();
        for (int i = 0; i < args.Length; i++)
        {
            var token = args[i];
            switch (token.ToLowerInvariant())
            {
                case "-h":
                case "--help":
                    r.ShowHelp = true;
                    break;
                default:
                    if (token.StartsWith("--", StringComparison.Ordinal))
                    {
                        string? value = null;
                        if (i + 1 < args.Length && !args[i + 1].StartsWith("--", StringComparison.Ordinal))
                        {
                            // Consume next as value only after verb+subverb have been parsed,
                            // so we never accidentally swallow a positional like the resource name.
                            if (r.Verb is not null && r.SubVerb is not null)
                            {
                                value = args[++i];
                            }
                        }
                        if (!r._options.TryGetValue(token, out var list))
                        {
                            list = new List<string?>();
                            r._options[token] = list;
                        }
                        list.Add(value);
                    }
                    else if (r.Verb is null)
                    {
                        r.Verb = token;
                    }
                    else if (r.SubVerb is null)
                    {
                        r.SubVerb = token;
                    }
                    else
                    {
                        r._positional.Add(token);
                    }
                    break;
            }
        }
        return r;
    }

    public string Required(int index, string name)
    {
        if (index >= _positional.Count)
        {
            throw new ArgumentException($"Missing required argument <{name}>.");
        }
        return _positional[index];
    }

    public string? GetOption(string name) =>
        _options.TryGetValue(name, out var v) ? v.LastOrDefault() : null;

    public IEnumerable<string> GetOptionAll(string name) =>
        _options.TryGetValue(name, out var v)
            ? v.Where(x => x is not null)!.Cast<string>()
            : Enumerable.Empty<string>();

    public bool GetBool(string name) => _options.ContainsKey(name);
}
